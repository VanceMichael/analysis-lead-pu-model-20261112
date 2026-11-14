// Feature extraction and preprocessing for the PU learner.
//
// Only attributes that are knowable at score_at may be features. Every record
// carries score_at (the scoring timestamp) and known_at (when its label was
// observed), so any label/timing metadata is excluded structurally:
//   - label, reviewProbability, id, company identify the row or its outcome;
//   - known_at reveals label observation timing.
// The policy's `features` list is an allowlist on top of that. This is the
// strongest visibility check the version-1 record schema supports: the data
// does not timestamp individual feature values, so callers must snapshot
// size/engagement/source at score_at. That assumption is written into the
// model artifact and the limitations.

export const FEATURE_SCHEMA = Object.freeze({
  size: "number",
  engagement: "number",
  source: "category",
  score_at: "integer"
});

export const FORBIDDEN_FEATURES = Object.freeze([
  "id", "company", "label", "reviewProbability", "known_at"
]);

export const DEFAULT_FEATURES = Object.freeze(["size", "engagement", "source"]);

export function resolveFeatureConfig(policy = {}) {
  const requested = policy.features ?? DEFAULT_FEATURES;
  if (!Array.isArray(requested) || requested.length === 0) {
    throw coded("invalidFeatures", "特征配置必须是非空字段数组");
  }
  if (new Set(requested).size !== requested.length) {
    throw coded("invalidFeatures", "特征配置不可重复");
  }
  for (const name of requested) {
    if (FORBIDDEN_FEATURES.includes(name)) {
      throw coded("forbiddenFeature", `字段 ${name} 不可作为特征（非 score_at 前可见属性）`);
    }
    if (!Object.hasOwn(FEATURE_SCHEMA, name)) {
      throw coded("unknownFeature", `未知特征字段 ${name}`);
    }
  }
  return { names: [...requested], types: { ...FEATURE_SCHEMA } };
}

export function coded(code, message, details) {
  const error = new Error(message);
  error.code = code;
  if (details !== undefined) error.details = details;
  return error;
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function validateFeatureValue(record, name, type) {
  const value = record[name];
  const valid = type === "integer" ? Number.isSafeInteger(value)
    : type === "number" ? isFiniteNumber(value) : typeof value === "string";
  if (!valid) {
    throw coded("invalidFeature", `字段 ${name} 的值不符合 ${type} 约定`, { feature: name });
  }
  return value;
}

// Fit category vocabularies, numeric moments and the design-name layout on
// training rows only. Everything needed to transform a future row is stored in
// this dictionary; `predict` never re-fits anything.
export function fitTransform(records, featureNames, { quadratic = true } = {}) {
  if (!Array.isArray(records) || records.length === 0) {
    throw coded("emptyTraining", "没有可用于拟合预处理字典的训练行");
  }
  const categories = {};
  const numeric = [];
  for (const name of featureNames) {
    const type = FEATURE_SCHEMA[name];
    if (type === "category") {
      categories[name] = [...new Set(records.map(row => String(row[name])))]
        .sort((left, right) => left.localeCompare(right, "en"));
      continue;
    }
    const values = records.map(row => validateFeatureValue(row, name, type));
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
      Math.max(1, values.length - 1);
    numeric.push({ name, mean, scale: variance > 0 ? Math.sqrt(variance) : 1 });
  }
  const designNames = [];
  for (const name of featureNames) {
    if (FEATURE_SCHEMA[name] === "category") {
      for (const value of categories[name]) designNames.push(`${name}=${value}`);
    } else {
      designNames.push(name);
      if (quadratic) designNames.push(`${name}^2`);
    }
  }
  return {
    version: 1,
    features: [...featureNames],
    types: { ...FEATURE_SCHEMA },
    categories,
    numeric,
    design: { quadratic, oneHot: true },
    designNames,
    unknownCategory: "reject"
  };
}

// Transform one record into the numeric design vector. Missing fields, wrong
// types, unseen categories and post-score metadata all fail with stable codes.
export function transformRecord(record, transform, { context = "记录" } = {}) {
  if (record === null || typeof record !== "object" || Array.isArray(record)) {
    throw coded("invalidRow", `${context} 不是对象`);
  }
  const missing = transform.features.filter(name => !Object.hasOwn(record, name));
  if (missing.length) {
    throw coded("missingColumn", `${context} 缺少字段：${missing.join(", ")}`, { missing });
  }
  const numeric = new Map(transform.numeric.map(item => [item.name, item]));
  const values = [];
  for (const name of transform.features) {
    const type = transform.types[name];
    if (type === "category") {
      const value = validateFeatureValue(record, name, type);
      const vocabulary = transform.categories[name];
      if (!vocabulary.includes(value)) {
        throw coded("unknownCategory",
          `${context} 的 ${name}=${value} 未出现在训练词典中，拒绝预测`, { feature: name, value });
      }
      for (const candidate of vocabulary) values.push(candidate === value ? 1 : 0);
      continue;
    }
    const value = validateFeatureValue(record, name, type);
    const { mean, scale } = numeric.get(name);
    const standardized = scale === 0 ? 0 : (value - mean) / scale;
    values.push(standardized);
    if (transform.design.quadratic) values.push(standardized ** 2);
  }
  return values;
}
