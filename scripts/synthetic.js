// Deterministic synthetic data with a controllable, known label-missingness
// mechanism. Used by tests and the demo; no randomness is hidden from the
// caller (the PRNG is seeded and reproducible).

export function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Scenario fields:
//   n                number of main records
//   sources          array of source names (default two sources)
//   prevalence       P(y=1) per source, either number or map
//   labelPropensity  e_s per source: P(observed positive | true positive)
//   negativeReview   P(observed negative | true negative) per source
//   scoreAt          constant score_at (all features knowable before it)
//   idPrefix         deterministic id prefix
//   signal           optional { sizeWeight, engagementWeight } shifting the
//                    log-odds of a true positive so the learner has something
//                    to find. Default 0 means pure noise (AUC ~0.5).
//
// Row generation:
//   logit(pi_row) = logit(pi_s) + signal.sizeWeight * z(size)
//                                     + signal.engagementWeight * z(engagement)
//   draw y ~ Bernoulli(pi_row)
//   draw u ~ Uniform
//   if y and u < e_s          -> label positive
//   if !y and u < neg_s       -> label negative
//   otherwise                 -> label unlabeled
// This is exactly the single-training-set mechanism, so e_s is known.
export function generateScenario(config) {
  const random = mulberry32(config.seed ?? 7);
  const sources = config.sources ?? ['expo', 'web'];
  const n = config.n ?? 120;
  const scoreAt = config.scoreAt ?? 40;
  const signal = config.signal ?? { sizeWeight: 0, engagementWeight: 0 };
  const records = [];
  const truth = [];
  const prevalenceOf = source =>
    typeof config.prevalence === 'number' ? config.prevalence : config.prevalence[source];
  const eOf = source => (typeof config.labelPropensity === 'number'
    ? config.labelPropensity : config.labelPropensity[source]);
  const negOf = source => {
    const value = config.negativeReview ?? 0.5;
    return typeof value === 'number' ? value : value[source];
  };
  const logit = p => Math.log(p / (1 - p));
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  // size in [10, 39], engagement in [0, 8]; standardise deterministically.
  const sizeZ = size => (size - 24.5) / 8.66;
  const engagementZ = engagement => (engagement - 4) / 2.58;
  for (let index = 0; index < n; index += 1) {
    const source = sources[index % sources.length];
    const pi = prevalenceOf(source);
    const e = eOf(source);
    const neg = negOf(source);
    const size = 10 + (index % 30);
    const engagement = index % 9;
    const rowLogit = logit(pi) + signal.sizeWeight * sizeZ(size) +
      signal.engagementWeight * engagementZ(engagement);
    const piRow = Math.min(0.999, Math.max(0.001, sigmoid(rowLogit)));
    const y = random() < piRow ? 1 : 0;
    const knownAt = config.scoreAt != null ? scoreAt : 0;
    let label;
    const draw = random();
    if (y === 1 && draw < e) label = 'positive';
    else if (y === 0 && draw < neg) label = 'negative';
    else label = 'unlabeled';
    records.push({
      id: `${config.idPrefix ?? 'g'}${String(index).padStart(4, '0')}`,
      company: `${config.idPrefix ?? 'g'}c${String(index).padStart(4, '0')}`,
      score_at: scoreAt,
      known_at: knownAt,
      source,
      size,
      engagement,
      label,
      reviewProbability: 0
    });
    truth.push({ id: records[index].id, y, source, label });
  }
  return { records, truth };
}

// An independent review sample with known inclusion probability. Every
// company is distinct from the main scenario so company-level isolation holds.
// `positive` is the true outcome (the review is trustworthy); probability is
// the independent inclusion probability used for inverse-probability weights.
export function generateAuditSample(config) {
  const random = mulberry32(config.seed ?? 11);
  const rows = [];
  const sources = config.sources ?? ['expo', 'web'];
  const n = config.n ?? 40;
  const probability = config.probability ?? 0.2;
  const signal = config.signal ?? { sizeWeight: 0, engagementWeight: 0 };
  const prevalenceOf = source =>
    typeof config.prevalence === 'number' ? config.prevalence : config.prevalence[source];
  const logit = p => Math.log(p / (1 - p));
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const sizeZ = size => (size - 24.5) / 8.66;
  const engagementZ = engagement => (engagement - 4) / 2.58;
  for (let index = 0; index < n; index += 1) {
    const source = sources[index % sources.length];
    const pi = prevalenceOf(source);
    const size = 10 + (index % 30);
    const engagement = index % 9;
    const piRow = Math.min(0.999, Math.max(0.001,
      sigmoid(logit(pi) + signal.sizeWeight * sizeZ(size) +
        signal.engagementWeight * engagementZ(engagement))));
    rows.push({
      company: `audit${String(index).padStart(4, '0')}`,
      size,
      engagement,
      source,
      positive: random() < piRow,
      probability,
      score_at: 40,
      known_at: 40
    });
  }
  return rows;
}
