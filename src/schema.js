export const schema = {
  "id": "string",
  "company": "string",
  "score_at": "integer",
  "known_at": "integer",
  "source": "string",
  "size": "number",
  "engagement": "number",
  "label": "string",
  "reviewProbability": "number"
};

export const LABELS = Object.freeze(["positive", "negative", "unlabeled"]);

export const MODEL_VERSION = 1;
export const ARTIFACT_VERSION = 1;

export const POLICY_DEFAULTS = Object.freeze({
  features: ["size", "engagement", "source"],
  capacity: 10,
  priorRange: [0.1, 0.4],
  auditSource: "reviewSample",
  scMechanism: "bernoulli",
  calibrationBins: 5
});
