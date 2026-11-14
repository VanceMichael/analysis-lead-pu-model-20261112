import test from 'node:test';
import assert from 'node:assert/strict';
import { fitPuModel } from '../src/pu.js';
import { generateScenario, generateAuditSample } from '../scripts/synthetic.js';

// Controlled label-missingness experiments. Each scenario has a known
// mechanism, so we can assert the required refusal / degradation behaviour:
//   all-unlabeled      -> refuse (cannot separate positive from unlabeled)
//   very few positives -> train but degrade to uncalibrated ranking
//   stratified conflict-> degrade, report per-source sensitivity
//   implausible prior  -> fall back to range, degrade
//   valid SC           -> calibrate
const SIGNAL = { sizeWeight: 1.5, engagementWeight: 0 };
const BASE = { prevalence: { expo: 0.3, web: 0.3 }, negativeReview: 0.2, signal: SIGNAL };

function auditFor(prevalence, n = 3000) {
  return generateAuditSample({ n, seed: 9, probability: 0.2, prevalence, signal: SIGNAL });
}

function policyFor(audit, overrides = {}) {
  return { features: ['size', 'engagement', 'source'], capacity: 5,
    priorRange: [0.1, 0.6], minPositives: 2, reviewSample: audit, ...overrides };
}

test('受控标签缺失：全未标注直接拒绝', () => {
  const { records } = generateScenario({ ...BASE, n: 200, seed: 1,
    labelPropensity: { expo: 0, web: 0 } });
  assert.equal(records.filter(row => row.label === 'positive').length, 0);
  assert.throws(() => fitPuModel(records, policyFor(auditFor(BASE.prevalence)), {}),
    error => error.code === 'noPositiveLabels');
});

test('受控标签缺失：正例极少时训练但降级为未校准排序', () => {
  const { records } = generateScenario({ ...BASE, n: 1200, seed: 2,
    labelPropensity: { expo: 0.008, web: 0.008 } });
  const out = fitPuModel(records, policyFor(auditFor(BASE.prevalence)), {});
  assert.equal(out.model.calibrated, false);
  assert.equal(out.model.degraded, true);
  assert.equal(out.diagnostics.assumptions.calibrationStable, false);
  assert.equal(out.model.calibrationDivisor, null);
  assert.ok(out.diagnostics.warnings.includes('degraded_to_ranking'));
});

test('受控标签缺失：分层倾向冲突降级并给出按来源敏感性', () => {
  const { records } = generateScenario({ ...BASE, n: 600, seed: 3,
    labelPropensity: { expo: 0.95, web: 0.03 } });
  const out = fitPuModel(records, policyFor(auditFor(BASE.prevalence)), {});
  assert.equal(out.model.calibrated, false);
  assert.ok(out.diagnostics.warnings.includes('sc_assumption_violated'));
  assert.ok(out.diagnostics.warnings.includes('prior_source_conflict'));
  assert.ok(out.model.labelingPropensity.spread > out.model.labelingPropensity.tolerance);
  assert.equal(out.model.sensitivity.perSource.length, 2);
  for (const source of out.model.sensitivity.perSource) {
    assert.ok(source.riskUnderPropensityRange.length > 0);
  }
});

test('受控标签缺失：先验越出业务范围时退回区间并降级', () => {
  const audit = auditFor({ expo: 0.9, web: 0.9 });
  const { records } = generateScenario({ ...BASE, n: 600, seed: 4,
    prevalence: { expo: 0.9, web: 0.9 }, labelPropensity: { expo: 0.85, web: 0.85 } });
  const out = fitPuModel(records, policyFor(audit, { priorRange: [0.05, 0.2] }), {});
  assert.equal(out.model.priorSelection.fallback, true);
  assert.equal(out.model.calibrated, false);
  assert.ok(out.diagnostics.warnings.includes('prior_out_of_policy_range'));
});

test('受控标签缺失：SC 成立时可以校准', () => {
  const { records } = generateScenario({ ...BASE, n: 800, seed: 6,
    labelPropensity: { expo: 0.85, web: 0.85 } });
  const out = fitPuModel(records, policyFor(auditFor(BASE.prevalence)), {});
  assert.equal(out.model.calibrated, true);
  assert.equal(out.model.degraded, false);
  assert.ok(out.model.calibrationDivisor > 0 && out.model.calibrationDivisor <= 1);
  assert.equal(out.diagnostics.warnings.length, 0);
});
