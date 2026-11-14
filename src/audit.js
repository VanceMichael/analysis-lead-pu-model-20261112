// Review-sample handling: validation, inverse-probability estimates of the
// class prior and of per-source label propensity.
//
// The review sample is an independently reviewed draw with known sampling
// probabilities. It is used only for (1) estimating the class prior / label
// propensity and (2) out-of-sample validation. It is never appended to the
// ordinary PU training rows. Weights are 1/probability (Horvitz-Thompson).

import { fingerprint } from './dataset.js';
import { coded } from './features.js';
import { weightedMean, wilsonInterval } from './stats.js';

export const AUDIT_FIELDS = Object.freeze({
  company: 'string', size: 'number', engagement: 'number', source: 'string',
  positive: 'boolean', probability: 'number', score_at: 'integer', known_at: 'integer'
});

// Validate the analysis configuration itself. A malformed policy must fail
// loudly at train time rather than quietly altering the prior or the regularisation.
export function validatePolicy(policy) {
  if (policy === null || typeof policy !== 'object' || Array.isArray(policy)) {
    throw coded('invalidPolicy', '业务配置必须是对象');
  }
  if (policy.priorRange !== undefined && policy.priorRange !== null) {
    const range = policy.priorRange;
    if (!Array.isArray(range) || range.length !== 2 ||
        !finite(range[0]) || !finite(range[1]) ||
        !(range[0] > 0) || !(range[1] < 1) || !(range[0] <= range[1])) {
      throw coded('invalidPolicy',
        'priorRange 必须是 (0,1) 内且 low <= high 的两个有限数');
    }
  }
  if (policy.capacity !== undefined && !Number.isSafeInteger(policy.capacity)) {
    throw coded('invalidPolicy', 'capacity 必须是整数');
  }
  for (const name of ['minPositives', 'minCalibrationPositives']) {
    if (policy[name] !== undefined && !Number.isSafeInteger(policy[name])) {
      throw coded('invalidPolicy', `${name} 必须是整数`);
    }
  }
  if (policy.l2 !== undefined && !(finite(policy.l2) && policy.l2 >= 0)) {
    throw coded('invalidPolicy', 'l2 必须是有限的非负数');
  }
  if (policy.scTolerance !== undefined &&
      !(finite(policy.scTolerance) && policy.scTolerance >= 0)) {
    throw coded('invalidPolicy', 'scTolerance 必须是有限的非负数');
  }
  if (policy.scMechanism !== undefined && policy.scMechanism !== 'bernoulli') {
    throw coded('invalidPolicy', `不支持的 scMechanism：${policy.scMechanism}`);
  }
  return policy;
}

function finite(value) { return typeof value === 'number' && Number.isFinite(value); }

export function validateAuditSample(rows) {
  if (!Array.isArray(rows)) throw coded('invalidAuditSample', 'reviewSample 必须是审阅记录数组');  for (const [index, row] of rows.entries()) {
    if (row === null || typeof row !== 'object' || Array.isArray(row)) {
      throw coded('invalidAuditSample', `reviewSample 第 ${index + 1} 行不是对象`);
    }
    for (const [field, type] of Object.entries(AUDIT_FIELDS)) {
      const value = row[field];
      const valid = type === 'boolean' ? typeof value === 'boolean'
        : type === 'integer' ? Number.isSafeInteger(value) : finite(value) || typeof value === type;
      if (!valid) {
        throw coded('invalidAuditSample',
          `reviewSample 第 ${index + 1} 行的 ${field} 不符合 ${type} 约定`);
      }
    }
    if (!(row.probability > 0 && row.probability <= 1)) {
      throw coded('invalidAuditSample', 'reviewSample 的 probability 必须位于 (0, 1]');
    }
    if (row.company.length === 0) throw coded('invalidAuditSample', 'reviewSample 的 company 为空');
  }
  return rows;
}

export function auditVersion(rows, policy) {
  return fingerprint({
    reviewSample: rows,
    priorRange: policy.priorRange ?? null,
    sampling: 'inverse_probability'
  });
}

export function assertCompanyDisjoint(trainRows, auditRows, testRows = []) {
  const trainCompanies = new Set(trainRows.map(row => row.company));
  const auditCompanies = new Set(auditRows.map(row => row.company));
  const leaked = [...auditCompanies].filter(company => trainCompanies.has(company));
  if (leaked.length) {
    throw coded('companyLeakage',
      `同一企业同时出现在训练与审阅验证侧：${leaked.slice(0, 5).join(', ')}`,
      { companies: leaked });
  }
  const testCompanies = new Set(testRows.map(row => row.company));
  const testLeaked = [...auditCompanies].filter(company => testCompanies.has(company));
  if (testLeaked.length) {
    throw coded('companyLeakage',
      `同一企业同时出现在测试与审阅验证侧：${testLeaked.slice(0, 5).join(', ')}`,
      { companies: testLeaked });
  }
  const byName = (left, right) => left.localeCompare(right, 'en');
  return {
    trainCompanies: [...trainCompanies].sort(byName),
    testCompanies: [...testCompanies].sort(byName),
    auditCompanies: [...auditCompanies].sort(byName)
  };
}

// Inverse-probability weighted prior. Returns per-source estimates and Wilson
// intervals; the interval width reflects unequal probability weights through
// the Kish effective sample size.
export function estimateAuditPrior(auditRows) {
  if (auditRows.length === 0) {
    return { available: false, global: null, perSource: [], totalRows: 0, weightedRows: 0 };
  }
  const weights = auditRows.map(row => 1 / row.probability);
  const positives = auditRows.map(row => (row.positive ? 1 : 0));
  const globalMean = weightedMean(positives, weights);
  const global = globalMean.estimate === null ? null : {
    estimate: globalMean.estimate,
    interval: wilsonInterval(globalMean.estimate, globalMean.effectiveN),
    effectiveN: globalMean.effectiveN,
    totalWeight: globalMean.totalWeight,
    weightedPositives: auditRows.reduce((sum, row) => sum + (row.positive ? 1 / row.probability : 0), 0),
    rows: auditRows.length,
    positives: auditRows.filter(row => row.positive).length
  };
  const sources = [...new Set(auditRows.map(row => row.source))].sort((a, b) => a.localeCompare(b, 'en'));
  const perSource = sources.map(source => {
    const rows = auditRows.filter(row => row.source === source);
    const localWeights = rows.map(row => 1 / row.probability);
    const localPositive = rows.map(row => (row.positive ? 1 : 0));
    const mean = weightedMean(localPositive, localWeights);
    return {
      source,
      rows: rows.length,
      weightedRows: mean.totalWeight,
      positives: rows.filter(row => row.positive).length,
      weightedPositives: rows.reduce((sum, row) => sum + (row.positive ? 1 / row.probability : 0), 0),
      estimate: mean.estimate,
      interval: mean.estimate === null ? null : wilsonInterval(mean.estimate, mean.effectiveN),
      effectiveN: mean.effectiveN
    };
  });
  return { available: true, global, perSource, totalRows: auditRows.length, weightedRows: weights.reduce((a, b) => a + b, 0) };
}

// The prior actually fed into PU weighting. If the audit estimate is
// degenerate or outside the business prior range, fall back to the configured
// range midpoint and mark the model degraded/uncalibrated rather than fitting
// with an unsupported weight.
export function choosePrior(auditPrior, policy) {
  const range = policy.priorRange ?? null;
  const withinRange = value => range === null || (value >= range[0] && value <= range[1]);
  if (auditPrior.available && auditPrior.global) {
    const estimate = auditPrior.global.estimate;
    if (Number.isFinite(estimate) && estimate > 0 && estimate < 1) {
      if (withinRange(estimate)) {
        return { source: 'audit_weighted_positive_rate', estimate, interval: auditPrior.global.interval,
          auditEstimate: estimate, fallback: false, withinRange: true, degenerate: false };
      }
      if (range) {
        return { source: 'policy_prior_range_fallback', estimate: (range[0] + range[1]) / 2,
          interval: { low: range[0], high: range[1] }, auditEstimate: estimate,
          fallback: true, withinRange: false, degenerate: false };
      }
      return { source: 'audit_weighted_positive_rate', estimate, interval: auditPrior.global.interval,
        auditEstimate: estimate, fallback: false, withinRange: true, degenerate: false };
    }
  }
  if (range) {
    return { source: 'policy_prior_range_fallback', estimate: (range[0] + range[1]) / 2,
      interval: { low: range[0], high: range[1] }, auditEstimate: auditPrior.global?.estimate ?? null,
      fallback: true, withinRange: true, degenerate: true };
  }
  return null;
}

// q_s = P(observed positive label | source) in the ordinary (non-audit) rows.
// Under single-training-set SC, q_s = e_s * pi_s, so e_s = q_s / pi_s. An
// e_s above 1 is logically impossible and proves the "positives are labeled
// at random" assumption false for that source.
export function estimateLabelPropensity(mainRows, auditPrior, policy) {
  const sources = [...new Set(mainRows.map(row => row.source))]
    .sort((a, b) => a.localeCompare(b, 'en'));
  const auditBySource = new Map(auditPrior.perSource.map(item => [item.source, item]));
  const perSource = sources.map(source => {
    const rows = mainRows.filter(row => row.source === source);
    const labelRate = rows.length === 0 ? null : rows.filter(row => row.label === 'positive').length / rows.length;
    const audit = auditBySource.get(source);
    const pi = audit?.estimate ?? null;
    const piInterval = audit?.interval ?? null;
    let e = null;
    let eInterval = null;
    let status = 'unknownPrior';
    if (labelRate !== null && pi !== null && pi > 0) {
      e = labelRate / pi;
      if (piInterval && piInterval.low > 0) {
        eInterval = { low: labelRate / piInterval.high, high: labelRate / piInterval.low };
      } else {
        eInterval = { low: null, high: null };
      }
      status = e > 1 + 1e-9 ? 'impossible' : e === 0 ? 'zeroLabeling' : 'possible';
      if (eInterval && Number.isFinite(eInterval.low) && eInterval.low > 1 + 1e-9) {
        // The entire interval sits above 1: not explainable by sampling noise.
        status = 'impossibleSignificant';
      }
    } else if (labelRate !== null) {
      status = pi === 0 ? 'impossibleZeroPrior' : 'unknownPrior';
    }
    return { source, rows: rows.length, labeledPositives: rows.filter(row => row.label === 'positive').length,
      labeledNegatives: rows.filter(row => row.label === 'negative').length,
      labelRate, prior: pi, priorInterval: piInterval, e, eInterval, status };
  });
  const finite = perSource.filter(item => item.e !== null && Number.isFinite(item.e));
  const spread = finite.length >= 2 ? Math.max(...finite.map(item => item.e)) -
    Math.min(...finite.map(item => item.e)) : 0;
  // A point estimate above 1 is logically incompatible with SC, but with small
  // samples it can be noise. We separate "flagged" from "significantly
  // impossible" (the whole interval above 1) and only degrade calibration on
  // the latter, or on a conflict between sources.
  const anyImpossible = finite.some(item => item.e > 1 + 1e-9);
  const impossibleSignificant = perSource.some(item =>
    item.eInterval && Number.isFinite(item.eInterval.low) && item.eInterval.low > 1 + 1e-9);
  const upperExceedsOne = perSource.some(item => item.eInterval && item.eInterval.high !== null &&
    item.eInterval.high > 1 + 1e-9);
  const conflict = spread > (policy.scTolerance ?? 0.25);

  // Pooled propensity under the (assumed) constant-e model. Because per-source
  // priors differ, pool with the prevalence-weighted row mass:
  //   e_pooled = sum_s labeled_s / sum_s pi_s * n_s.
  const totalRows = perSource.reduce((sum, item) => sum + item.rows, 0);
  const totalLabeled = perSource.reduce((sum, item) => sum + item.labeledPositives, 0);
  const expectedPositiveMass = perSource.reduce((sum, item) =>
    sum + (item.prior !== null ? item.prior * item.rows : 0), 0);
  const knownPriorMass = perSource.every(item => item.prior !== null);
  const pooledE = knownPriorMass && expectedPositiveMass > 0
    ? totalLabeled / expectedPositiveMass : null;
  // Interval from the aggregate audit prior interval: e in [q/pi_high, q/pi_low].
  let pooledEInterval = null;
  if (pooledE !== null) {
    const weightedHigh = perSource.reduce((sum, item) =>
      sum + (item.rows * (item.priorInterval?.high ?? item.prior ?? 1)), 0);
    const weightedLow = perSource.reduce((sum, item) =>
      sum + (item.rows * (item.priorInterval?.low ?? item.prior ?? 1)), 0);
    pooledEInterval = {
      low: weightedHigh > 0 ? totalLabeled / weightedHigh : null,
      high: weightedLow > 0 ? totalLabeled / weightedLow : null
    };
  }
  return {
    concept: 'e(source) = P(observed positive label | source) / P(true positive | source)',
    source: 'main labeled-positive rate divided by audit prior',
    scAssumption: '正例随机被标注（single-training-set SC）',
    calibration: 'SC 下辅助分数估计 P(已标注正例|x)=e·P(y=1|x)，故 P(y=1|x)=g(x)/e；' +
      '除以类先验 π 只有当 e=1 时才成立',
    perSource,
    pooled: { e: pooledE, interval: pooledEInterval, labeledPositives: totalLabeled,
      expectedPositiveMass, rows: totalRows },
    spread,
    tolerance: policy.scTolerance ?? 0.25,
    conflict,
    anyImpossible,
    impossibleSignificant,
    upperExceedsOne,
    plausible: !impossibleSignificant && !conflict
  };
}
