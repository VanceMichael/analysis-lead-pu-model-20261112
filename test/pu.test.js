import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fitPuModel, splitByCompany, scoreRecord, buildReviewQueue, limitations } from '../src/pu.js';
import { generateScenario, generateAuditSample } from '../scripts/synthetic.js';
import { resolveFeatureConfig } from '../src/features.js';
import { validateAuditSample, estimateAuditPrior } from '../src/audit.js';
import { evaluateOnAudit, requireConfirmedNegatives } from '../src/evaluate.js';
import { weightedAuc, wilsonInterval } from '../src/stats.js';

const SIGNAL = { sizeWeight: 1.5, engagementWeight: 0 };
const BASE = {
  n: 600, seed: 5,
  prevalence: { expo: 0.3, web: 0.3 },
  labelPropensity: { expo: 0.8, web: 0.8 },
  negativeReview: 0.2,
  signal: SIGNAL
};

function policyOf(auditConfig, overrides = {}) {
  const audit = generateAuditSample({
    n: 2000, seed: 3, probability: 0.2, signal: SIGNAL,
    prevalence: auditConfig.prevalence ?? BASE.prevalence
  });
  return {
    features: ['size', 'engagement', 'source'],
    capacity: 10,
    priorRange: [0.1, 0.6],
    minPositives: 2,
    reviewSample: audit,
    ...overrides
  };
}

test('PU 训练把 positive / negative / unlabeled 严格区分', () => {
  const { records } = generateScenario(BASE);
  const out = fitPuModel(records, policyOf(BASE), {});
  // Confirmed negatives are never fit labels and never validation labels.
  assert.equal(out.model.assumptions.negativesExcludedFromFit, true);
  assert.equal(out.model.assumptions.auditNotUsedAsTrainingRows, true);
  // The fit universe is labeled positives + unlabeled only.
  const trainRows = out.diagnostics.counts.train;
  assert.equal(out.model.fitted.rows, trainRows.positive + trainRows.unlabeled);
  assert.ok(trainRows.negative > 0, '场景应包含确认负例，用于检查其被排除');
});

test('审阅样本不混入训练，且与训练/测试企业不相交', () => {
  const { records } = generateScenario(BASE);
  const policy = policyOf(BASE);
  const auditCompanies = new Set(policy.reviewSample.map(row => row.company));
  const out = fitPuModel(records, policy, {});
  // Audit companies are distinct from every main company by construction.
  for (const row of records) assert.equal(auditCompanies.has(row.company), false);
  assert.equal(out.diagnostics.split.companyDisjointTrainTest, true);
  assert.equal(out.diagnostics.split.auditDisjointTrain, true);
  assert.equal(out.diagnostics.split.auditDisjointTest, true);
  // Every main company is assigned to exactly one side of the split.
  const trainSet = new Set(out.diagnostics.split.disjoint.trainCompanies);
  const testSet = new Set(out.diagnostics.split.disjoint.testCompanies);
  const mainCompanies = new Set(records.map(row => row.company));
  for (const company of mainCompanies) {
    assert.equal(trainSet.has(company) !== testSet.has(company), true);
  }
  assert.equal(trainSet.size + testSet.size, mainCompanies.size);
});

test('企业级切分不共享企业，且不受行序影响', () => {
  const { records } = generateScenario(BASE);
  const forward = splitByCompany(records, 0.25);
  const reverse = splitByCompany([...records].reverse(), 0.25);
  assert.deepEqual(forward.trainCompanies, reverse.trainCompanies);
  const overlap = forward.trainCompanies.filter(company => forward.testCompanies.includes(company));
  assert.deepEqual(overlap, []);
});

test('先验与标注倾向的来源被记录', () => {
  const { records } = generateScenario(BASE);
  const out = fitPuModel(records, policyOf(BASE), {});
  assert.equal(out.model.priorSelection.source, 'audit_weighted_positive_rate');
  assert.match(out.diagnostics.labelPropensity.source, /标注正例率|labeled/i);
  assert.ok(out.diagnostics.auditPrior.global.estimate > 0);
  assert.ok(out.model.labelingPropensity.pooled > 0);
  assert.ok(out.model.labelingPropensity.interval);
  // Under a valid SC scenario the model is allowed to calibrate.
  assert.equal(out.model.calibrated, true);
  assert.ok(out.model.calibrationDivisor > 0 && out.model.calibrationDivisor <= 1);
});

test('正例随机标注假设成立时可校准，分数带风险解释', () => {
  const { records } = generateScenario({ ...BASE, labelPropensity: { expo: 0.85, web: 0.85 } });
  const out = fitPuModel(records, policyOf(BASE), {});
  assert.equal(out.model.degraded, false);
  const auditRow = out.model;
  const scored = scoreRecord(out.model, records[0]);
  assert.equal(scored.calibrated, true);
  assert.ok(scored.risk !== null && scored.risk >= 0 && scored.risk <= 1);
  assert.equal(out.model.riskFunction.kind, 'calibrated_positive_risk');
  assert.match(out.model.riskFunction.interpretation, /不是真实成交概率/);
  assert.ok(out.model.prior > 0);
});

test('全为未标注时拒绝训练而不是把未标注当负例', () => {
  const { records } = generateScenario({ ...BASE, n: 100, labelPropensity: { expo: 0, web: 0 } });
  assert.equal(records.filter(row => row.label === 'unlabeled').length > 0, true);
  assert.equal(records.filter(row => row.label === 'positive').length, 0);
  assert.throws(() => fitPuModel(records, policyOf(BASE), {}),
    error => error.code === 'noPositiveLabels');
});

test('正例极少时拒绝或降级为未校准排序', () => {
  const { records } = generateScenario({ ...BASE, n: 800, labelPropensity: { expo: 0.01, web: 0.01 } });
  const positives = records.filter(row => row.label === 'positive').length;
  assert.ok(positives > 0 && positives < 10);
  const out = fitPuModel(records, policyOf(BASE), {});
  assert.equal(out.model.calibrated, false);
  assert.equal(out.model.degraded, true);
  assert.equal(out.diagnostics.assumptions.calibrationStable, false);
  const scored = scoreRecord(out.model, records[0]);
  assert.equal(scored.risk, null);
  assert.equal(scored.calibrated, false);
});

test('来源层标注倾向冲突时降级并给出按来源敏感性范围', () => {
  const { records } = generateScenario({
    ...BASE, labelPropensity: { expo: 0.95, web: 0.03 }
  });
  const out = fitPuModel(records, policyOf(BASE), {});
  assert.equal(out.model.calibrated, false);
  assert.equal(out.model.degraded, true);
  assert.ok(out.diagnostics.warnings.includes('prior_source_conflict'));
  assert.ok(out.model.labelingPropensity.conflict);
  // Per-source sensitivity range is present and conceptually distinct.
  assert.equal(out.model.sensitivity.perSource.length, 2);
  for (const item of out.model.sensitivity.perSource) {
    assert.ok(item.labelPropensity >= 0);
    assert.ok(item.labelPropensityInterval);
  }
  assert.equal(out.model.sensitivity.divisorKind, 'labeling_propensity');
  // Uncalibrated score must not be presented as a probability.
  const scored = scoreRecord(out.model, records.find(row => row.label === 'unlabeled'));
  assert.equal(scored.risk, null);
  assert.ok(limitations(out.model).some(item => /不是成交概率|未校准/.test(item)));
});

test('先验不合理时拒绝或降级，不默默采用', () => {
  // Audit says ~0.9 prevalence but policy only allows [0.1, 0.6].
  const tiny = generateAuditSample({ n: 400, seed: 4, probability: 0.2,
    prevalence: { expo: 0.9, web: 0.9 }, signal: SIGNAL });
  const policy = { ...policyOf(BASE), priorRange: [0.05, 0.2], reviewSample: tiny };
  const { records } = generateScenario({ ...BASE });
  const out = fitPuModel(records, policy, {});
  assert.equal(out.model.priorSelection.fallback, true);
  assert.equal(out.model.calibrated, false);
  assert.ok(out.diagnostics.warnings.includes('prior_out_of_policy_range'));
});

test('没有审阅样本和先验范围时拒绝校准', () => {
  const { records } = generateScenario(BASE);
  const policy = { features: ['size', 'engagement', 'source'], capacity: 10,
    minPositives: 2, reviewSample: [] };
  const out = fitPuModel(records, policy, {});
  assert.equal(out.model.calibrated, false);
  assert.ok(out.diagnostics.warnings.includes('no_audit_sample'));
  assert.ok(out.diagnostics.warnings.includes('no_prior'));
  assert.equal(out.model.calibrationDivisor, null);
});

test('审阅样本按抽样概率加权，概率必须有效', () => {
  const audit = [
    { company: 'x1', size: 1, engagement: 0, source: 's', positive: true, probability: 0.1, score_at: 1, known_at: 1 },
    { company: 'x2', size: 2, engagement: 1, source: 's', positive: false, probability: 0.5, score_at: 1, known_at: 1 }
  ];
  const prior = estimateAuditPrior(audit);
  // Weighted estimate = (10) / (10 + 2) = 0.8333, not the unweighted 0.5.
  assert.ok(Math.abs(prior.global.estimate - 10 / 12) < 1e-12);
  assert.throws(() => validateAuditSample([{ ...audit[0], probability: 0 }]), /probability/);
  assert.throws(() => validateAuditSample([{ ...audit[0], probability: 1.5 }]), /probability/);
  assert.throws(() => validateAuditSample([{ ...audit[0], positive: 'yes' }]), /boolean/);
});

test('普通未标注行不能用于真负率', () => {
  const { records } = generateScenario(BASE);
  const unlabeled = records.filter(row => row.label === 'unlabeled');
  assert.throws(() => requireConfirmedNegatives(unlabeled),
    error => error.code === 'tnrRequiresConfirmedNegatives');
  const out = fitPuModel(records, policyOf(BASE), {});
  const evaluation = evaluateOnAudit(out.model, policyOf(BASE).reviewSample, { capacity: 10 });
  assert.equal(evaluation.trueNegativeRate, null);
  assert.equal(evaluation.unlabeledRowsUsedForTnr, false);
  // Confirmed-negative diagnostic exists but declares itself non-population.
  assert.equal(out.diagnostics.confirmedNegativeDiagnostic.populationEstimate, false);
});

test('特征禁止 label / reviewProbability / id / company / known_at', () => {
  for (const name of ['label', 'reviewProbability', 'id', 'company', 'known_at']) {
    assert.throws(() => resolveFeatureConfig({ features: ['size', name] }), /不可作为特征/);
  }
  assert.throws(() => resolveFeatureConfig({ features: ['size', 'nope'] }), /未知特征字段/);
  assert.deepEqual(resolveFeatureConfig({}).names, ['size', 'engagement', 'source']);
});

test('回访队列按企业计数、容量有界且确定', () => {
  const { records } = generateScenario(BASE);
  const out = fitPuModel(records, policyOf(BASE), {});
  const queueA = buildReviewQueue(out.model, records.filter(row => row.label === 'unlabeled'), { capacity: 5 });
  const queueB = buildReviewQueue(out.model, [...records].reverse().filter(row => row.label === 'unlabeled'), { capacity: 5 });
  assert.equal(queueA.queue.length, 5);
  assert.equal(new Set(queueA.queue.map(item => item.company)).size, 5);
  assert.deepEqual(queueA.queue, queueB.queue);
  for (const item of queueA.queue) assert.equal(item.action, 'human_review');
});

test('预测遇到缺列和未见类别行为一致', () => {
  const { records } = generateScenario(BASE);
  const out = fitPuModel(records, policyOf(BASE), {});
  const missing = { ...records[0] };
  delete missing.size;
  assert.throws(() => scoreRecord(out.model, missing), error => error.code === 'missingColumn');
  assert.throws(() => scoreRecord(out.model, { ...records[0], source: 'brand-new' }),
    error => error.code === 'unknownCategory');
});

test('配置字段非法时拒绝训练', () => {
  const { records } = generateScenario(BASE);
  const base = policyOf(BASE);
  const rejects = [
    { ...base, priorRange: [0.5, 0.1] },
    { ...base, priorRange: [-0.1, 0.5] },
    { ...base, priorRange: [0.1, 1.5] },
    { ...base, l2: -1 },
    { ...base, scMechanism: 'unknown' },
    { ...base, capacity: 2.5 }
  ];
  for (const policy of rejects) {
    assert.throws(() => fitPuModel(records, policy, {}),
      error => error.code === 'invalidPolicy', JSON.stringify(policy.priorRange));
  }
});

test('加权 AUC 在两类都缺失时不编造值', () => {
  assert.equal(weightedAuc([0.1, 0.2], [1, 1], [1, 1]), null);
  assert.equal(weightedAuc([0.1], [0], [1]), null);
});

test('Wilson 区间在无有效样本时退化为全范围', () => {
  assert.deepEqual(wilsonInterval(0.5, 0), { low: 0, high: 1 });
});
