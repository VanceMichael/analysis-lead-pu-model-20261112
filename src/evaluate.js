// Validation on the independent review sample.
//
// The review sample has known inclusion probabilities, so every metric is
// inverse-probability weighted (Horvitz-Thompson). Its rows are never part of
// fitting and never enter the ordinary training set.
//
// Reported metrics:
//   precision / recall over audited rows, weighted (with intervals)
//   positive coverage within the business review capacity
//   AUC as a ranking diagnostic (weighted, ties handled)
// Explicitly NOT reported: true-negative rate from unlabeled rows. An
// unlabeled row is not a confirmed negative, so a TNR computed on ordinary
// unlabeled rows would be fabricated. TNR is only defined on confirmed
// negatives and is labelled as a selection-biased diagnostic when shown.

import { scoreRecord } from './pu.js';
import { weightedMean, wilsonInterval, weightedAuc } from './stats.js';
import { coded } from './features.js';

function weightedCount(rows, weights, predicate) {
  return rows.reduce((sum, row, index) => sum + (predicate(row) ? weights[index] : 0), 0);
}

export function evaluateOnAudit(model, auditRows, { capacity, threshold = 0.5 } = {}) {
  if (!Array.isArray(auditRows) || auditRows.length === 0) {
    return { available: false, reason: '没有审阅验证样本', unlabeledRowsUsedForTnr: false };
  }
  try {
    const scored = auditRows.map((row, index) => {
      const { score, risk, calibrated } = scoreRecord(model, row, { context: `审阅第 ${index + 1} 行` });
      return { row, score, risk, calibrated, index };
    });
    const weights = auditRows.map(row => 1 / row.probability);
    const positives = auditRows.map(row => (row.positive ? 1 : 0));
    // Threshold-based precision/recall is only meaningful for a calibrated
    // risk. For an uncalibrated rank there is no probability threshold, so we
    // do not invent one; the capacity-limited block below is the business
    // metric that is valid either way.
    const useRisk = model.calibrated && scored.every(item => item.risk !== null);
    const valueOf = item => (useRisk ? item.risk : item.score);
    const predictedPositive = item => (useRisk ? item.risk >= threshold : null);

    const weightedPositives = weightedCount(auditRows, weights, row => row.positive);
    const weightedPredictedPositive = useRisk
      ? weightedCount(scored, weights, item => predictedPositive(item)) : null;
    const weightedTruePositive = useRisk
      ? weightedCount(scored, weights, item => item.row.positive && predictedPositive(item)) : null;
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    const effectiveN = (totalWeight * totalWeight) / weights.reduce((sum, w) => sum + w * w, 0);

    // Capacity-limited selection: weight of positive audits that fall inside
    // the top-`capacity` companies by score. This is the business question
    // ("can human review capacity cover positives?") and it is valid for both
    // calibrated and uncalibrated models.
    const companies = [...new Set(auditRows.map(row => row.company))];
    const companyScore = new Map();
    for (const item of scored) {
      const current = companyScore.get(item.row.company);
      const value = valueOf(item);
      if (current === undefined || value > current) companyScore.set(item.row.company, value);
    }
    const rankedCompanies = [...companyScore.entries()].sort((left, right) =>
      right[1] - left[1] || left[0].localeCompare(right[0], 'en'));
    const effectiveCapacity = Math.min(capacity ?? companies.length, companies.length);
    const selectedCompanies = new Set(rankedCompanies.slice(0, effectiveCapacity).map(([company]) => company));
    const positiveMass = weightedCount(auditRows, weights, row => row.positive);
    const coveredMass = weightedCount(scored, weights, item => item.row.positive && selectedCompanies.has(item.row.company));
    const coverageEstimate = positiveMass > 0 ? coveredMass / positiveMass : null;
    const selectedMass = weightedCount(scored, weights, item => selectedCompanies.has(item.row.company));
    const capacityPrecision = selectedMass > 0 ? coveredMass / selectedMass : null;

    // Precision/recall: threshold-based only when calibrated; otherwise use the
    // capacity-limited selection, which is still a legitimate weighted
    // estimate. The `basis` field states which definition was used.
    const thresholdPrecision = useRisk && weightedPredictedPositive > 0
      ? weightedTruePositive / weightedPredictedPositive : null;
    const thresholdRecall = useRisk && weightedPositives > 0
      ? weightedTruePositive / weightedPositives : null;
    const precisionEstimate = useRisk ? thresholdPrecision : capacityPrecision;
    const recallEstimate = useRisk ? thresholdRecall : coverageEstimate;
    const precisionDenominator = useRisk ? weightedPredictedPositive : selectedMass;
    const recallDenominator = useRisk ? weightedPositives : positiveMass;

    const precision = precisionEstimate === null ? null : {
      estimate: precisionEstimate,
      interval: wilsonInterval(precisionEstimate, effectiveN),
      basis: useRisk ? 'threshold' : 'capacity_selection',
      weightedDenominator: precisionDenominator
    };
    const recall = recallEstimate === null ? null : {
      estimate: recallEstimate,
      interval: wilsonInterval(recallEstimate, effectiveN),
      basis: useRisk ? 'threshold' : 'capacity_selection',
      weightedDenominator: recallDenominator
    };

    const auc = weightedAuc(
      scored.map(item => valueOf(item)),
      positives,
      weights
    );

    return {
      available: true,
      scoredAs: useRisk ? 'calibrated_risk' : 'uncalibrated_rank',
      threshold: useRisk ? threshold : null,
      weighting: 'inverse_probability',
      rows: auditRows.length,
      weightedRows: totalWeight,
      effectiveN,
      positives: auditRows.filter(row => row.positive).length,
      weightedPositives: positiveMass,
      precision,
      recall,
      precisionNote: useRisk
        ? '按校准风险阈值 0.5 的加权精确率'
        : '未校准排序无概率阈值，精确率按回访容量内选中企业的加权正例比例计算',
      capacity: effectiveCapacity,
      capacityPrecision: capacityPrecision === null ? null : {
        estimate: capacityPrecision,
        interval: wilsonInterval(capacityPrecision, effectiveN),
        weightedNumerator: coveredMass,
        weightedDenominator: selectedMass,
        note: '回访容量内选中企业的加权正例比例'
      },
      positiveCoverageInCapacity: coverageEstimate === null ? null : {
        estimate: coverageEstimate,
        interval: wilsonInterval(coverageEstimate, effectiveN),
        weightedNumerator: coveredMass,
        weightedDenominator: positiveMass
      },
      auc: auc === null ? null : { estimate: auc, weighting: 'inverse_probability' },
      trueNegativeRate: null,
      unlabeledRowsUsedForTnr: false,
      tnrNote: '真负率只在确认负例上定义；普通未标注行不用于计算真负率。',
      calibration: useRisk ? '按外部标注倾向换算，仅供排序与敏感性分析' : '未校准'
    };
  } catch (error) {
    if (error.code === 'unknownCategory' || error.code === 'missingColumn') {
      // An audit row outside the training vocabulary must not silently be
      // imputed; report the validation as unavailable with the reason.
      return { available: false, reason: `审阅验证无法评分：${error.message}`, code: error.code,
        unlabeledRowsUsedForTnr: false };
    }
    throw error;
  }
}

// Selection-biased diagnostic only. Confirmed negatives come from completed
// reviews, so any rate on them is not a population estimate.
export function confirmedNegativeDiagnostic(model, negativeRows, { threshold = 0.5 } = {}) {
  if (!negativeRows || negativeRows.length === 0) {
    return { available: false, reason: '没有确认负例', populationEstimate: false };
  }
  const scored = negativeRows.map((row, index) =>
    scoreRecord(model, row, { context: `确认负例第 ${index + 1} 行` }));
  const useRisk = model.calibrated && scored.every(item => item.risk !== null);
  const flagged = scored.filter(item => (useRisk ? item.risk : item.score) >= threshold).length;
  return {
    available: true,
    rows: scored.length,
    flagged: flagged / scored.length,
    basis: useRisk ? 'calibrated_risk_at_threshold' : 'uncalibrated_rank_at_threshold',
    note: '确认负例来自既有回访选择，此比例只作偏倚诊断，不代表总体真负率。',
    populationEstimate: false
  };
}

// Guard: a true-negative rate is only defined on confirmed negatives. Any
// caller that passes ordinary unlabeled rows is rejected rather than silently
// producing a fabricated number.
export function requireConfirmedNegatives(rows) {
  if (!Array.isArray(rows) || rows.some(row => row?.label !== 'negative')) {
    throw coded('tnrRequiresConfirmedNegatives',
      '真负率只能在确认负例上计算，普通未标注行不可用作负例');
  }
  return rows;
}

