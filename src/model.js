// Model artifact persistence.
//
// One artifact file holds, so they can never drift apart:
//   - the fitted coefficients and the exact preprocessing dictionary
//     (feature list, category vocabulary, numeric moments, design layout);
//   - the assumption configuration (SC mechanism, prior range, tolerance,
//     capacity, minPositives, l2);
//   - the review-sample version/hash and input fingerprint;
//   - prior selection provenance and the per-source sensitivity envelope.
//
// `predict` loads one artifact and applies exactly the stored transform; it
// never re-fits or silently defaults a missing column/category.

import { fingerprint } from './dataset.js';
import { scoreRecord } from './pu.js';
import { coded } from './features.js';
import { MODEL_VERSION, ARTIFACT_VERSION } from './schema.js';

export { ARTIFACT_VERSION };

export function buildArtifact(model, { policy, datasetFingerprint, reviewVersion, fittedAt = null }) {
  const assumptions = {
    scMechanism: policy.scMechanism ?? 'bernoulli',
    scTolerance: policy.scTolerance ?? 0.25,
    priorRange: policy.priorRange ?? null,
    capacity: policy.capacity ?? 10,
    minPositives: policy.minPositives ?? 3,
    minCalibrationPositives: policy.minCalibrationPositives ?? 10,
    l2: Number.isFinite(policy.l2) ? policy.l2 : 1,
    features: policy.features ?? null,
    scoreVisibility: '仅 score_at 前可见属性；label/reviewProbability/id/company/known_at 禁止'
  };
  return {
    artifactVersion: ARTIFACT_VERSION,
    model,
    preprocessing: model.features,
    assumptions,
    reviewSampleVersion: reviewVersion,
    datasetFingerprint,
    fittedAt,
    integrity: fingerprint({
      model, preprocessing: model.features, assumptions,
      reviewSampleVersion: reviewVersion, datasetFingerprint, fittedAt
    })
  };
}

export function validateArtifact(artifact) {
  if (artifact === null || typeof artifact !== 'object') throw coded('invalidArtifact', '模型文件不是对象');
  if (artifact.artifactVersion !== ARTIFACT_VERSION) {
    throw coded('unsupportedArtifact', `模型文件版本 ${artifact.artifactVersion} 不支持`);
  }
  if (!artifact.model || artifact.model.modelVersion !== MODEL_VERSION) {
    throw coded('unsupportedModel', '模型内容版本不支持');
  }
  if (!artifact.preprocessing || !Array.isArray(artifact.preprocessing.features)) {
    throw coded('invalidArtifact', '模型缺少预处理字典');
  }
  const expected = fingerprint({
    model: artifact.model, preprocessing: artifact.preprocessing,
    assumptions: artifact.assumptions, reviewSampleVersion: artifact.reviewSampleVersion,
    datasetFingerprint: artifact.datasetFingerprint, fittedAt: artifact.fittedAt
  });
  if (expected !== artifact.integrity) {
    throw coded('artifactIntegrity', '模型文件完整性校验失败：内容与保存时不一致');
  }
  return artifact;
}

// Predict one row against an already-validated artifact. Callers scoring many
// rows should call validateArtifact once (or use validateArtifact themselves)
// and pass `{ validated: true }`; the default re-checks for safety when the
// artifact comes straight from disk.
export function predictRow(artifact, record, { context = '预测行', validated = false } = {}) {
  if (!validated) validateArtifact(artifact);
  const result = scoreRecord(artifact.model, record, { context });
  const calibrated = artifact.model.calibrated && result.risk !== null;
  return {
    ...result,
    scoreKind: calibrated ? 'positive_risk_estimate' : 'uncalibrated_rank',
    interpretation: calibrated
      ? '假设下的正例风险，非真实成交概率'
      : '仅排序；未校准，不解释为概率',
    warnings: modelWarnings(artifact.model)
  };
}

export function modelWarnings(model) {
  const warnings = [];
  if (model.degraded) warnings.push('degraded_to_ranking');
  if (model.labelPropensity?.conflict) warnings.push('prior_source_conflict');
  if (model.labelPropensity?.anyImpossible) warnings.push('label_propensity_impossible');
  if (!model.calibrated) warnings.push('uncalibrated');
  return warnings;
}

// Load-time compatibility check against a dataset: refuse if the data the
// artifact was fit on cannot be matched by its preprocessing dictionary.
export function assertCompatible(artifact, records) {
  validateArtifact(artifact);
  const features = artifact.preprocessing.features;
  for (const [index, row] of records.entries()) {
    const missing = features.filter(name => !Object.hasOwn(row, name));
    if (missing.length) {
      throw coded('missingColumn', `第 ${index + 1} 行缺少字段：${missing.join(', ')}`, { missing });
    }
  }
  return true;
}
