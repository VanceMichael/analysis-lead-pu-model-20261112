// Positive-Unlabeled (PU) risk learner.
//
// Label semantics (kept strictly separate):
//   positive  - confirmed positive, usable as a labeled positive.
//   negative  - confirmed failure from a completed review. It is NOT the same
//               as "unlabeled"; these rows are selection-biased review
//               outcomes, so they are excluded from fitting and from any
//               true-negative-rate estimate on ordinary rows.
//   unlabeled - not yet reviewed. It is a mixture of true positives and true
//               negatives. It is neither a negative nor a validation label.
//
// Estimator (Elkan/Noto-style non-negative PU risk, with an explicit prior):
//   g(x) is an auxiliary scorer fitted to separate labeled positives from
//   unlabeled rows. The external class prior pi comes from the independent
//   review sample (inverse-probability weighted) or a configured range; it is
//   never re-estimated from the unlabeled rows. Under the single-training-set
//   "positives are labeled at random" assumption, g(x) estimates
//   P(observed positive label | x) = e * P(y=1|x), where the labeling
//   propensity e = P(observed positive | true positive) is estimated per
//   source from the labeled-positive rate and the audit prior. The positive
//   risk is therefore r(x) = clip(g(x) / e, 0, 1), NOT g(x)/pi (those coincide
//   only when e = 1). When the assumption fails or e is not identified, the
//   artifact is marked uncalibrated and only the rank is reported.
//
// All fitted state (design dictionary, category vocabulary, numeric moments,
// coefficients, prior, labeling propensity, assumption config, audit version,
// sensitivity) lives in one portable JSON artifact.

import { fitTransform, transformRecord, coded, resolveFeatureConfig } from './features.js';
import { fitLogistic, linearScore, sigmoid, clampProbability } from './stats.js';
import {
  validateAuditSample, auditVersion, assertCompanyDisjoint, estimateAuditPrior,
  choosePrior, estimateLabelPropensity, validatePolicy
} from './audit.js';
import { evaluateOnAudit, confirmedNegativeDiagnostic } from './evaluate.js';
import { MODEL_VERSION } from './schema.js';

export { MODEL_VERSION };

function stableKey(company) {
  let hash = 2166136261;
  for (let index = 0; index < company.length; index += 1) {
    hash ^= company.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function designOf(records, transform) {
  return records.map((row, index) => [1, ...transformRecord(row, transform, {
    context: `训练第 ${index + 1} 行`
  })]);
}

function checkLabelVocabulary(records) {
  const allowed = new Set(['positive', 'negative', 'unlabeled']);
  for (const [index, row] of records.entries()) {
    if (!allowed.has(row.label)) {
      throw coded('unknownLabel', `第 ${index + 1} 行的 label=${row.label} 不在 positive/negative/unlabeled 内`);
    }
  }
}

function countsByLabel(rows) {
  return {
    rows: rows.length,
    positive: rows.filter(row => row.label === 'positive').length,
    negative: rows.filter(row => row.label === 'negative').length,
    unlabeled: rows.filter(row => row.label === 'unlabeled').length
  };
}

// Company-disjoint split. Companies are ordered by a stable hash so that
// record order and id naming cannot decide the split. No company may appear
// on both sides.
//
// A purely random company split can strand every labeled positive on the test
// side, which would look like "there are no positives" when the real issue is
// the split. We therefore try a small sequence of deterministic salts and
// keep the first split that leaves enough labeled positives (and unlabeled
// rows) in training. The choice is deterministic and recorded. Companies are
// still never shared between sides; only which whole companies go where
// changes.
function splitWithSalt(records, fraction, salt) {
  const companies = [...new Set(records.map(row => row.company))]
    .sort((left, right) => stableKey(`${salt}:${left}`) - stableKey(`${salt}:${right}`) ||
      left.localeCompare(right, 'en'));
  if (companies.length < 4) {
    throw coded('tooFewCompanies', `企业数 ${companies.length} 不足以做企业级切分`);
  }
  const holdoutCount = Math.max(1, Math.min(companies.length - 1, Math.round(companies.length * fraction)));
  const holdout = new Set(companies.slice(0, holdoutCount));
  const train = records.filter(row => !holdout.has(row.company));
  const test = records.filter(row => holdout.has(row.company));
  const trainCompanies = new Set(train.map(row => row.company));
  const testCompanies = new Set(test.map(row => row.company));
  const overlap = [...trainCompanies].filter(company => testCompanies.has(company));
  if (overlap.length) throw coded('companyLeakage', `切分后企业重叠：${overlap.join(', ')}`);
  return {
    salt, train, test,
    trainCompanies: [...trainCompanies].sort((a, b) => a.localeCompare(b, 'en')),
    testCompanies: [...testCompanies].sort((a, b) => a.localeCompare(b, 'en'))
  };
}

export function splitByCompany(records, fraction = 0.25, salt = 'pu-split-v1', { requiredTrainPositives = 1 } = {}) {
  const totalPositives = records.filter(row => row.label === 'positive').length;
  const target = Math.min(requiredTrainPositives, totalPositives);
  let first = null;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const candidate = splitWithSalt(records, fraction, `${salt}#${attempt}`);
    if (first === null) first = candidate;
    const trainPositives = candidate.train.filter(row => row.label === 'positive').length;
    const trainUnlabeled = candidate.train.filter(row => row.label === 'unlabeled').length;
    if (trainPositives >= target && (trainUnlabeled > 0 || totalPositives === records.length)) {
      return candidate;
    }
  }
  return first;
}

function buildTransformRows(positiveRows, unlabeledRows) {
  return [...positiveRows, ...unlabeledRows];
}

export function fitPuModel(records, policy = {}, options = {}) {
  validatePolicy(policy);
  checkLabelVocabulary(records);
  if (records.length === 0) throw coded('emptyDataset', '没有记录');
  const features = resolveFeatureConfig(policy);
  const auditRows = validateAuditSample(policy.reviewSample ?? []);
  const capacity = policy.capacity ?? 10;
  if (!Number.isSafeInteger(capacity) || capacity < 1) throw coded('invalidCapacity', 'capacity 必须是正整数');
  const minPositives = policy.minPositives ?? 3;
  if (!Number.isSafeInteger(minPositives) || minPositives < 1) {
    throw coded('invalidPolicy', 'minPositives 必须是正整数');
  }
  const minCalibrationPositives = policy.minCalibrationPositives ?? 10;
  if (!Number.isSafeInteger(minCalibrationPositives) || minCalibrationPositives < 1) {
    throw coded('invalidPolicy', 'minCalibrationPositives 必须是正整数');
  }

  const totalPositives = records.filter(row => row.label === 'positive').length;
  const split = options.split ?? splitByCompany(records, options.holdoutFraction ?? 0.25,
    'pu-split-v1', { requiredTrainPositives: Math.min(minPositives, totalPositives) });
  const disjoint = assertCompanyDisjoint(split.train, auditRows, split.test);
  const positiveRows = split.train.filter(row => row.label === 'positive');
  const unlabeledRows = split.train.filter(row => row.label === 'unlabeled');
  const negativeRows = split.train.filter(row => row.label === 'negative');
  if (positiveRows.length === 0) {
    throw coded('noPositiveLabels',
      '训练侧没有已确认正例；无法区分正例与未标注，拒绝训练');
  }
  if (positiveRows.length < minPositives) {
    throw coded('tooFewPositiveLabels',
      `训练侧已确认正例仅 ${positiveRows.length} 条，少于下限 ${minPositives}；` +
      'PU 先验与标注倾向不可识别，拒绝训练（可将 minPositives 调低以强制降级试算）');
  }
  if (unlabeledRows.length === 0) {
    throw coded('noUnlabeledRows', '训练侧没有未标注行；PU 风险无法估计，拒绝训练');
  }

  const auditPrior = estimateAuditPrior(auditRows);
  const propensity = estimateLabelPropensity(split.train, auditPrior, policy);
  const priorSelection = choosePrior(auditPrior, policy);

  // Fit the design dictionary on the PU universe only. Confirmed negatives do
  // not define the feature space because they are not a random sample.
  const transform = fitTransform(buildTransformRows(positiveRows, unlabeledRows), features.names);
  const positiveDesign = designOf(positiveRows, transform);
  const unlabeledDesign = designOf(unlabeledRows, transform);
  const design = [...positiveDesign, ...unlabeledDesign];
  const targets = [...positiveDesign.map(() => 1), ...unlabeledDesign.map(() => 0)];
  const weights = new Array(design.length).fill(1);
  const fitted = fitLogistic(design, targets, weights,
    { l2: Number.isFinite(policy.l2) ? policy.l2 : 1 });

  const priorUsable = priorSelection !== null &&
    Number.isFinite(priorSelection.estimate) && priorSelection.estimate > 0 &&
    priorSelection.estimate < 1;
  const pooledE = propensity.pooled?.e ?? null;
  const pooledInterval = propensity.pooled?.interval ?? null;
  // Under SC, g(x) estimates P(observed positive label | x) = e * P(y=1|x), so
  // the calibrated positive risk is g(x)/e. Calibration requires e to be a
  // valid probability (0 < e <= 1). The interval is kept as an uncertainty
  // band and reported in the sensitivity block rather than pretending the
  // point estimate is exact.
  const eIdentified = Number.isFinite(pooledE) && pooledE > 0 && pooledE <= 1;
  const positiveMass = propensity.pooled?.labeledPositives ??
    positiveRows.length;
  const assumptions = assessAssumptions(priorSelection, propensity, priorUsable, eIdentified,
    { labeledPositives: positiveMass, minCalibrationPositives });
  const calibrated = priorUsable && assumptions.plausible;
  const degraded = !calibrated;
  const calibrationDivisor = calibrated ? pooledE : null;
  const sensitivity = buildSensitivity(propensity, policy, unlabeledDesign, fitted.beta);
  const riskFunction = {
    kind: calibrated ? 'calibrated_positive_risk' : 'uncalibrated_rank',
    calibrated,
    formula: calibrated ? 'clip(sigmoid(beta·x) / e, 0, 1)' : 'sigmoid(beta·x)',
    divisor: calibrated ? { kind: 'labeling_propensity', value: pooledE,
      source: 'audit_prior_and_main_label_rate' } : null,
    interpretation: calibrated
      ? '在 SC 与先验假设下估计的正例风险；仍不是真实成交概率'
      : '仅保证排序；分数未经校准，不得解释为成交概率或正例概率'
  };

  const model = {
    modelVersion: MODEL_VERSION,
    kind: 'positive-unlabeled-risk',
    features: transform,
    coefficients: fitted.beta,
    interceptIndex: 0,
    prior: priorUsable ? priorSelection.estimate : null,
    priorSelection,
    calibrationDivisor,
    labelingPropensity: {
      pooled: pooledE,
      interval: pooledInterval,
      pooledDivisorUsable: eIdentified,
      spread: propensity.spread, tolerance: propensity.tolerance,
      conflict: propensity.conflict, anyImpossible: propensity.anyImpossible,
      perSource: propensity.perSource, concept: propensity.concept,
      calibration: propensity.calibration
    },
    calibrated,
    degraded,
    riskFunction,
    assumptions: {
      sc: 'single-training-set：在每个来源层内，正例被标注的概率与特征无关',
      negativesExcludedFromFit: true,
      auditNotUsedAsTrainingRows: true,
      priorExternalToUnlabeled: true,
      divisorIsLabelingPropensity: true,
      featuresVisibleAtScoreAt: features.names
    },
    sensitivity,
    fitted: { converged: fitted.converged, rows: design.length, designWidth: transform.designNames.length + 1 },
    datasetFingerprint: options.datasetFingerprint ?? null,
    auditVersion: auditVersion(auditRows, policy),
    policyFingerprint: options.policyFingerprint ?? null
  };

  const evaluation = auditRows.length
    ? evaluateOnAudit(model, auditRows, { capacity })
    : { available: false, reason: '没有审阅验证样本', unlabeledRowsUsedForTnr: false };
  const negativeDiagnostic = confirmedNegativeDiagnostic(model, negativeRows);

  const queue = buildReviewQueue(model, split.test.filter(row => row.label === 'unlabeled'), {
    capacity, ranking: degraded ? 'balanced_source' : 'global'
  });

  return {
    model,
    diagnostics: {
      split: {
        companies: {
          total: new Set(records.map(row => row.company)).size,
          train: split.trainCompanies.length,
          test: split.testCompanies.length
        },
        trainRows: split.train.length,
        testRows: split.test.length,
        auditRows: auditRows.length,
        disjoint: disjoint,
        companyDisjointTrainTest: true,
        auditDisjointTrain: true,
        auditDisjointTest: true
      },
      counts: { train: countsByLabel(split.train), test: countsByLabel(split.test) },
      excludedFromFit: {
        confirmedNegatives: negativeRows.length,
        reason: '确认负例来自既有回访选择，不作为 PU 拟合标签',
        auditRows: auditRows.length,
        auditReason: '审阅样本只用于先验估计与验证，不并入普通训练行'
      },
      auditPrior,
      priorSelection,
      labelPropensity: propensity,
      assumptions,
      evaluation,
      confirmedNegativeDiagnostic: negativeDiagnostic,
      warnings: buildWarnings(assumptions, priorSelection, propensity, auditPrior, degraded)
    },
    queue
  };
}

function assessAssumptions(priorSelection, propensity, priorUsable, eIdentified,
  { labeledPositives, minCalibrationPositives }) {
  const reasons = [];
  if (!priorUsable) reasons.push('没有可用的外部先验（审阅样本或 priorRange）');
  if (priorSelection?.fallback) {
    reasons.push('审阅先验估计落在业务允许范围之外，已退回区间中点');
  }
  if (!eIdentified) {
    reasons.push('合并标注倾向 e 不可用作除数（非正或超过 1）');
  }
  const unstable = !(labeledPositives >= minCalibrationPositives);
  if (unstable) {
    reasons.push(`训练侧已标注正例仅 ${labeledPositives} 条，少于校准所需 ${minCalibrationPositives}；` +
      '标注倾向估计不稳定，降级为未校准排序');
  }
  if (propensity.anyImpossible) {
    reasons.push('至少一个来源的标注倾向 e 点估计超过 1，按定义与“正例随机被标注”矛盾');
  }
  if (propensity.impossibleSignificant) {
    reasons.push('某来源标注倾向 e 的区间下界超过 1，标注机制无法用 SC 解释');
  }
  if (propensity.conflict) {
    reasons.push(`各来源标注倾向极差 ${propensity.spread.toFixed(4)} 超过容差 ${propensity.tolerance}`);
  }
  if (propensity.upperExceedsOne) {
    reasons.push('标注倾向的置信上界超过 1，说明该来源的标注机制无法识别');
  }
  return {
    plausible: priorUsable && !priorSelection?.fallback && eIdentified &&
      !unstable && !propensity.impossibleSignificant && !propensity.anyImpossible &&
      !propensity.conflict,
    priorUsable,
    eIdentified,
    calibrationStable: !unstable,
    labeledPositives,
    minCalibrationPositives,
    scViolated: propensity.anyImpossible || propensity.impossibleSignificant || propensity.conflict,
    reasons
  };
}

function buildWarnings(assumptions, priorSelection, propensity, auditPrior, degraded) {
  const warnings = [];
  if (!auditPrior.available) warnings.push('no_audit_sample');
  if (assumptions.scViolated) warnings.push('sc_assumption_violated');
  if (propensity.conflict) warnings.push('prior_source_conflict');
  if (propensity.anyImpossible) warnings.push('label_propensity_impossible');
  if (priorSelection?.fallback) warnings.push('prior_out_of_policy_range');
  if (priorSelection === null) warnings.push('no_prior');
  if (degraded) warnings.push('degraded_to_ranking');
  return warnings;
}

function summarise(values) {
  if (values.length === 0) return { rows: 0, mean: null, low: null, high: null };
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = q => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
  return {
    rows: sorted.length,
    mean: Number((values.reduce((a, b) => a + b, 0) / values.length).toFixed(6)),
    low: Number(quantile(0.1).toFixed(6)),
    high: Number(quantile(0.9).toFixed(6))
  };
}

// Sensitivity envelope. Without the SC assumption there is no single
// identifiable divisor, so we report the per-source range: for each source we
// divide the auxiliary score by that source's labeling-propensity interval
// (equivalently by its prior interval times a constant e) and summarise the
// resulting risk distribution. This is the required "按来源层的敏感性范围".
function buildSensitivity(propensity, policy, unlabeledDesign, beta) {
  const gValues = unlabeledDesign.map(row => clampProbability(sigmoid(linearScore(row, beta))));

  const byDivisor = (label, divisor) => ({
    label,
    divisor: Number(divisor.toFixed(6)),
    riskSummary: divisor > 0 && divisor <= 1
      ? summarise(gValues.map(g => Math.min(1, Math.max(0, g / divisor))))
      : { rows: gValues.length, mean: null, low: null, high: null, note: '除数不可用于校准' }
  });

  const globalBounds = [];
  if (policy.priorRange) {
    // A policy range on the prior does not by itself define e; under the
    // e=1 special case it is the degenerate divisor range. Record it explicitly
    // as a prior range, not as a probability claim.
    globalBounds.push({ label: 'priorRange.low', prior: policy.priorRange[0] },
      { label: 'priorRange.high', prior: policy.priorRange[1] });
  }
  if (propensity.pooled?.e != null) {
    globalBounds.push({ label: 'pooledLabelingPropensity', prior: propensity.pooled.e });
  }

  const perSource = propensity.perSource.map(item => {
    const interval = item.eInterval ?? { low: item.e, high: item.e };
    const divisorBounds = [];
    if (Number.isFinite(interval.low) && interval.low > 0) {
      divisorBounds.push(byDivisor(`${item.source}.e.low`, interval.low));
    }
    if (Number.isFinite(interval.high) && interval.high > 0) {
      divisorBounds.push(byDivisor(`${item.source}.e.high`, interval.high));
    }
    return {
      source: item.source,
      rows: item.rows,
      labelRate: item.labelRate,
      prior: item.prior,
      priorInterval: item.priorInterval,
      labelPropensity: item.e,
      labelPropensityInterval: item.eInterval,
      status: item.status,
      riskUnderPropensityRange: divisorBounds
    };
  });

  return {
    unit: 'unlabeled_training_rows',
    divisorKind: 'labeling_propensity',
    globalDivisorBounds: globalBounds.map(item => ({
      label: item.label,
      value: Number(item.prior.toFixed(6)),
      note: item.label.startsWith('priorRange') ? '业务先验范围，不是标注倾向' : undefined
    })),
    perSource,
    note: 'SC 不成立时以来源层的标注倾向区间给出敏感性范围；分数不是成交概率。'
  };
}

export function scoreRecord(model, record, { context = '预测行' } = {}) {
  if (!model || model.modelVersion !== MODEL_VERSION) {
    throw coded('unsupportedModel', '模型版本不支持');
  }
  const design = [1, ...transformRecord(record, model.features, { context })];
  const g = clampProbability(sigmoid(linearScore(design, model.coefficients)));
  const divisor = model.calibrationDivisor;
  const calibrated = model.calibrated && Number.isFinite(divisor) && divisor > 0 && divisor <= 1;
  const risk = calibrated ? Math.min(1, Math.max(0, g / divisor)) : null;
  return { score: Number(g.toFixed(12)), risk, calibrated };
}

// Human review queue. Capacity counts companies, not rows, so a company with
// several leads never consumes several review slots. When the model is
// degraded (SC assumption violated or prior unusable) the queue ranks within
// source and interleaves sources: the global auxiliary score is confounded by
// which source happened to be reviewed, so a global top-k would keep chasing
// the already-favoured source.
export function buildReviewQueue(model, records, { capacity, ranking = 'global' } = {}) {
  if (!Number.isSafeInteger(capacity) || capacity < 1) throw coded('invalidCapacity', 'capacity 必须是正整数');
  if (!Array.isArray(records)) throw coded('invalidQueueInput', '回访队列输入必须是记录数组');
  if (records.length === 0) {
    return { capacity, ranking, candidates: 0, queue: [],
      note: '没有待回访的未标注记录；队列为空。' };
  }
  const identity = (record, index) => ({
    // groupKey is the business review unit: a company when present, otherwise
    // the record itself. This keeps "one slot per company" true when leads are
    // scored fresh without a company column.
    groupKey: typeof record.company === 'string' && record.company.length > 0
      ? record.company : `row:${record.id ?? index}`,
    company: typeof record.company === 'string' && record.company.length > 0
      ? record.company : null,
    id: record.id ?? null
  });
  const scored = records.map((record, index) => {
    const { score, risk, calibrated } = scoreRecord(model, record, { context: `回访队列第 ${index + 1} 行` });
    return { record, score, risk, calibrated, ...identity(record, index), index };
  });
  const valueOf = item => (item.risk ?? item.score);
  const byScore = (left, right) => valueOf(right) - valueOf(left) || right.score - left.score ||
    left.groupKey.localeCompare(right.groupKey, 'en') ||
    String(left.id ?? '').localeCompare(String(right.id ?? ''), 'en') ||
    left.index - right.index;
  const useBalanced = ranking === 'balanced_source' && model.features.features.includes('source');
  let ordered;
  if (useBalanced) {
    const groups = new Map();
    for (const item of scored) {
      if (!groups.has(item.record.source)) groups.set(item.record.source, []);
      groups.get(item.record.source).push(item);
    }
    for (const group of groups.values()) group.sort(byScore);
    const decorated = [];
    for (const [source, group] of groups) {
      group.forEach((item, index) => decorated.push({
        ...item, percentile: (group.length - index) / group.length, source
      }));
    }
    decorated.sort((left, right) => right.percentile - left.percentile || byScore(left, right));
    ordered = decorated;
  } else {
    ordered = [...scored].sort(byScore);
  }
  const queue = [];
  const seen = new Set();
  for (const item of ordered) {
    if (queue.length >= capacity) break;
    if (seen.has(item.groupKey)) continue;
    seen.add(item.groupKey);
    queue.push({
      rank: queue.length + 1,
      id: item.id,
      company: item.company,
      source: item.record.source,
      score: Number(item.score.toFixed(6)),
      risk: item.risk === null ? null : Number(item.risk.toFixed(6)),
      scoreKind: item.calibrated ? 'positive_risk_estimate' : 'uncalibrated_rank',
      action: 'human_review'
    });
  }
  return {
    capacity,
    ranking: useBalanced ? 'balanced_source' : 'global',
    candidates: scored.length,
    queue,
    note: '按模型排序的人工回访候选；不是成交概率，也不替代销售决策。' +
      (useBalanced ? '因标注机制冲突，排序按来源层内分位交叉，避免单一来源占满容量。' : '')
  };
}

export function limitations(model) {
  const items = [
    '分数仅用于安排人工回访排序，不是成交概率，也不应直接作为销售结论。',
    '特征只使用 score_at 前可取的属性（本数据快照中为 size/engagement/source）；' +
      '若上游在 score_at 之后回填属性，本模型无法检测。',
    '普通未标注行不作为负例，也不用于计算真负率；确认负例不参与拟合。',
    '已确认负例来自销售既有回访选择，存在选择偏倚，因此模型不把“未回访”当成“失败”。'
  ];
  if (model.degraded) {
    items.push('SC“正例随机被标注”假设不成立或先验不可用：输出已降级为未校准排序，' +
      'risk 字段为 null，不得解释为概率。');
    items.push('请按 artifact.sensitivity.perSource 的来源层敏感性范围理解不确定性，' +
      '并用新一轮分层审阅补足数据。');
  } else {
    items.push('即使校准，r(x) 是假设下的正例风险，不是真实成交概率；' +
      '来源层抽样概率变化会改变先验，应重新审阅。');
  }
  if (model.labelPropensity?.conflict) {
    items.push('各来源标注倾向冲突：部分来源的正例几乎总被标注，另一些几乎从不被标注，' +
      '不能跨来源直接比较原始分数。');
  }
  return items;
}
