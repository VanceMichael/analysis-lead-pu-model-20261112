// Small, dependency-free numerical helpers used by the PU pipeline.
// Everything here is deterministic: no randomness, no wall-clock input.

export function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const tail = Math.exp(z);
  return tail / (1 + tail);
}

export function clampProbability(value) {
  return Math.min(1 - 1e-12, Math.max(1e-12, value));
}

export function logit(probability) {
  const p = clampProbability(probability);
  return Math.log(p / (1 - p));
}

// Gaussian elimination with partial pivoting. Throws on a singular system so a
// degenerate model fails loudly instead of returning NaNs.
export function solve(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-14) throw new Error("数值求解失败：矩阵奇异");
    if (pivot !== column) [a[pivot], a[column]] = [a[column], a[pivot]];
    const divisor = a[column][column];
    for (let row = column + 1; row < n; row += 1) {
      const factor = a[row][column] / divisor;
      if (factor === 0) continue;
      for (let index = column; index <= n; index += 1) a[row][index] -= factor * a[column][index];
    }
  }
  const solution = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let sum = a[row][n];
    for (let column = row + 1; column < n; column += 1) sum -= a[row][column] * solution[column];
    solution[row] = sum / a[row][row];
  }
  return solution;
}

// Weighted L2-regularised logistic regression fitted by damped Newton steps.
// The intercept is column 0 and is never penalised.
export function fitLogistic(design, targets, weights, options = {}) {
  const { l2 = 1, maxIterations = 200, tolerance = 1e-8 } = options;
  const rows = design.length;
  if (rows === 0) throw new Error("没有可用于拟合分类器的行");
  const width = design[0].length;
  if (design.some(row => row.length !== width)) throw new Error("设计矩阵列数不一致");
  let beta = new Array(width).fill(0);
  if (width > 0) beta[0] = Math.log(Math.max(1e-6, targets.reduce((s, y, i) => s + weights[i] * y, 0)) /
    Math.max(1e-6, targets.reduce((s, y, i) => s + weights[i] * (1 - y), 0)));
  let previous = -Infinity;
  let converged = false;
  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const probabilities = new Array(rows);
    const effective = new Array(rows);
    let objective = 0;
    for (let i = 0; i < rows; i += 1) {
      const row = design[i];
      let score = 0;
      for (let j = 0; j < width; j += 1) score += row[j] * beta[j];
      const p = clampProbability(sigmoid(score));
      probabilities[i] = p;
      effective[i] = weights[i] * p * (1 - p);
      const target = targets[i];
      objective += weights[i] * (target * Math.log(p) + (1 - target) * Math.log(1 - p));
    }
    for (let j = 1; j < width; j += 1) objective -= (l2 / 2) * beta[j] * beta[j];
    const gradient = new Array(width).fill(0);
    for (let j = 0; j < width; j += 1) {
      let sum = 0;
      for (let i = 0; i < rows; i += 1) sum += design[i][j] * weights[i] * (probabilities[i] - targets[i]);
      gradient[j] = sum + (j === 0 ? 0 : l2 * beta[j]);
    }
    if (Math.max(...gradient.map(Math.abs)) < tolerance) { converged = true; break; }
    const hessian = Array.from({ length: width }, () => new Array(width).fill(0));
    for (let i = 0; i < rows; i += 1) {
      const row = design[i];
      const w = effective[i];
      if (w === 0) continue;
      for (let j = 0; j < width; j += 1) {
        for (let k = j; k < width; k += 1) hessian[j][k] += w * row[j] * row[k];
      }
    }
    for (let j = 0; j < width; j += 1) {
      for (let k = 0; k < j; k += 1) hessian[j][k] = hessian[k][j];
      if (j > 0) hessian[j][j] += l2;
      hessian[j][j] += 1e-10;
    }
    const step = solve(hessian, gradient.map(value => -value));
    let scale = 1;
    let nextObjective = -Infinity;
    let nextBeta = beta;
    for (let attempt = 0; attempt < 30; attempt += 1) {
      nextBeta = beta.map((value, index) => value + scale * step[index]);
      nextObjective = 0;
      for (let i = 0; i < rows; i += 1) {
        const row = design[i];
        let score = 0;
        for (let j = 0; j < width; j += 1) score += row[j] * nextBeta[j];
        const p = clampProbability(sigmoid(score));
        nextObjective += weights[i] * (targets[i] * Math.log(p) + (1 - targets[i]) * Math.log(1 - p));
      }
      for (let j = 1; j < width; j += 1) nextObjective -= (l2 / 2) * nextBeta[j] * nextBeta[j];
      if (Number.isFinite(nextObjective) && nextObjective >= objective - 1e-12) break;
      scale /= 2;
    }
    beta = nextBeta;
    if (!Number.isFinite(nextObjective)) throw new Error("数值求解失败：目标函数非有限");
    if (Math.max(...step.map(value => Math.abs(value))) < tolerance) { converged = true; break; }
    if (Math.abs(nextObjective - previous) < tolerance) { converged = true; break; }
    previous = nextObjective;
  }
  return { beta, converged };
}

export function linearScore(row, beta) {
  let score = 0;
  for (let j = 0; j < beta.length; j += 1) score += row[j] * beta[j];
  return score;
}

// Wilson score interval for a weighted Bernoulli mean. `effectiveN` is the
// Kish effective sample size of the weights, so unequal sampling probabilities
// widen the interval instead of pretending every row is independent.
export function weightedMean(rows, weights) {
  let total = 0;
  let sumSquares = 0;
  let mass = 0;
  for (let i = 0; i < rows.length; i += 1) {
    total += weights[i];
    sumSquares += weights[i] * weights[i];
    mass += weights[i] * rows[i];
  }
  if (total === 0) return { estimate: null, effectiveN: 0, totalWeight: 0 };
  return { estimate: mass / total, effectiveN: (total * total) / sumSquares, totalWeight: total };
}

export function wilsonInterval(estimate, effectiveN, z = 1.959963984540054) {
  if (!(effectiveN > 0)) return { low: 0, high: 1 };
  const n = effectiveN;
  const denominator = 1 + (z * z) / n;
  const center = (estimate + (z * z) / (2 * n)) / denominator;
  const margin = (z / denominator) * Math.sqrt(estimate * (1 - estimate) / n + (z * z) / (4 * n * n));
  return { low: Math.max(0, center - margin), high: Math.min(1, center + margin) };
}

// Weighted ROC-AUC with rank-based tie handling. Returns null when either
// class has zero weight, rather than fabricating a value.
export function weightedAuc(scores, labels, weights) {
  const positiveWeight = labels.reduce((sum, y, i) => sum + (y ? weights[i] : 0), 0);
  const negativeWeight = labels.reduce((sum, y, i) => sum + (y ? 0 : weights[i]), 0);
  if (positiveWeight === 0 || negativeWeight === 0) return null;
  const order = scores.map((score, index) => ({ score, index }))
    .sort((left, right) => left.score - right.score || left.index - right.index);
  let satisfying = 0;
  let cumulativeNegative = 0;
  let index = 0;
  let aucNumerator = 0;
  while (index < order.length) {
    let end = index;
    while (end < order.length && order[end].score === order[index].score) end += 1;
    let tiePositive = 0;
    let tieNegative = 0;
    for (let cursor = index; cursor < end; cursor += 1) {
      const { index: row } = order[cursor];
      if (labels[row]) tiePositive += weights[row];
      else tieNegative += weights[row];
    }
    aucNumerator += tiePositive * (cumulativeNegative + tieNegative / 2);
    cumulativeNegative += tieNegative;
    satisfying += tiePositive;
    index = end;
  }
  return aucNumerator / (positiveWeight * negativeWeight);
}
