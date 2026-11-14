import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { loadDataset, fingerprint } from './dataset.js';
import { summarize } from './domain.js';
import { writeResult } from './output.js';
import { fitPuModel, buildReviewQueue, limitations } from './pu.js';
import { validateAuditSample, auditVersion } from './audit.js';
import { buildArtifact, predictRow, validateArtifact } from './model.js';

const COMMAND_OPTIONS = {
  summary: ['input', 'out'],
  inspect: ['input', 'out'],
  train: ['input', 'policy', 'out', 'report'],
  predict: ['input', 'model', 'out'],
  queue: ['input', 'model', 'out', 'capacity']
};

export function parseArguments(args) {
  const [command = 'summary', ...options] = args;
  if (!Object.hasOwn(COMMAND_OPTIONS, command)) {
    throw new Error('命令应为 summary、inspect、train、predict 或 queue');
  }
  const result = { command, input: 'data/records.json' };
  if (command === 'train') result.policy = 'data/policy.json';
  const allowed = COMMAND_OPTIONS[command];
  const assigned = new Set();
  for (let index = 0; index < options.length; index += 2) {
    const name = options[index];
    const value = options[index + 1];
    const key = typeof name === 'string' && name.startsWith('--') ? name.slice(2) : null;
    if (!key || !allowed.includes(key) || !value || value.startsWith('--')) {
      throw new Error(`参数应为 ${allowed.map(option => `--${option} 路径`).join(' 或 ')}`);
    }
    if (assigned.has(key) && key !== 'input') throw new Error('参数不可重复');
    assigned.add(key);
    result[key] = value;
  }
  if (['predict', 'queue'].includes(command) && !result.model) {
    throw new Error('predict/queue 需要 --model 路径');
  }
  return result;
}

export async function loadPolicy(path) {
  const policy = JSON.parse(await readFile(path, 'utf8'));
  if (policy === null || typeof policy !== 'object' || Array.isArray(policy)) {
    throw new Error('业务配置必须是对象');
  }
  return policy;
}

// Prediction input is validated by the model's preprocessing dictionary, not
// by the training schema. Missing columns and unseen categories therefore
// surface from the same transform path (`missingColumn` / `unknownCategory`),
// giving predict and queue one consistent rejection behavior. Only identity
// fields needed by the caller are checked; feature columns are deliberately
// not checked here so they fail through the same transform path.
export async function loadPredictionInput(path) {
  const document = JSON.parse(await readFile(path, 'utf8'));
  if (document.version !== 1) throw new Error('不支持的数据文件版本');
  if (!Array.isArray(document.records)) throw new Error('数据必须是记录数组');
  const ids = new Set();
  for (const [index, record] of document.records.entries()) {
    if (record === null || typeof record !== 'object' || Array.isArray(record)) {
      throw new Error(`第 ${index + 1} 条记录不是对象`);
    }
    if (Object.hasOwn(record, 'id')) {
      if (typeof record.id !== 'string' || record.id.length === 0) {
        throw new Error(`第 ${index + 1} 条记录的 id 为空或类型不是 string`);
      }
      if (ids.has(record.id)) throw new Error('记录 id 为空或重复');
      ids.add(record.id);
    }
  }
  return {
    version: 1,
    records: document.records,
    fingerprint: fingerprint([...document.records].sort((left, right) =>
      String(left.id ?? '').localeCompare(String(right.id ?? ''), 'en')))
  };
}

export async function main(args) {
  const options = parseArguments(args);
  const needsModel = options.command === 'predict' || options.command === 'queue';
  const dataset = needsModel ? await loadPredictionInput(options.input)
    : await loadDataset(options.input);
  let result;
  let written = false;
  if (options.command === 'summary') {
    result = summarize(dataset.records);
  } else if (options.command === 'inspect') {
    result = {
      records: dataset.records.length,
      fingerprint: dataset.fingerprint,
      fields: dataset.records.length ? Object.keys(dataset.records[0]).sort() : []
    };
  } else if (options.command === 'train') {
    result = await trainCommand(options, dataset);
    written = true;
  } else if (options.command === 'predict') {
    result = await predictCommand(options, dataset);
  } else {
    result = await queueCommand(options, dataset);
  }
  const document = { ...result, inputFingerprint: dataset.fingerprint };
  if (options.out && !written) await writeResult(options.out, document);
  return document;
}

async function trainCommand(options, dataset) {
  const policy = await loadPolicy(options.policy);
  const auditRows = validateAuditSample(policy.reviewSample ?? []);
  const policyFingerprint = fingerprint(policy);
  const fitted = fitPuModel(dataset.records, policy, {
    datasetFingerprint: dataset.fingerprint,
    policyFingerprint
  });
  const artifact = buildArtifact(fitted.model, {
    policy,
    datasetFingerprint: dataset.fingerprint,
    reviewVersion: auditVersion(auditRows, policy)
  });
  if (options.out) await writeResult(options.out, artifact);
  const report = {
    command: 'train',
    modelVersion: artifact.model.modelVersion,
    artifactVersion: artifact.artifactVersion,
    artifactIntegrity: artifact.integrity,
    reviewSampleVersion: artifact.reviewSampleVersion,
    policyFingerprint,
    calibrated: artifact.model.calibrated,
    degraded: artifact.model.degraded,
    prior: artifact.model.prior,
    priorSelection: artifact.model.priorSelection,
    assumptions: artifact.model.assumptions,
    diagnostics: fitted.diagnostics,
    sensitivity: artifact.model.sensitivity,
    evaluation: fitted.diagnostics.evaluation,
    queue: fitted.queue,
    limitations: limitations(artifact.model)
  };
  if (options.report) await writeResult(options.report, report);
  return report;
}

async function loadArtifact(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function predictCommand(options, dataset) {
  const artifact = validateArtifact(await loadArtifact(options.model));
  const predictions = dataset.records.map((record, index) => {
    const scored = predictRow(artifact, record, { context: `第 ${index + 1} 行`, validated: true });
    return { id: record.id ?? null, company: record.company ?? null, ...scored };
  });
  return {
    command: 'predict',
    modelIntegrity: artifact.integrity,
    calibrated: artifact.model.calibrated,
    warnings: artifact.model.degraded ? ['degraded_to_ranking'] : [],
    rows: predictions.length,
    predictions,
    limitations: limitations(artifact.model)
  };
}

async function queueCommand(options, dataset) {
  const artifact = validateArtifact(await loadArtifact(options.model));
  const capacity = options.capacity === undefined ? (artifact.assumptions?.capacity ?? 10)
    : Number(options.capacity);
  if (!Number.isSafeInteger(capacity) || capacity < 1) throw new Error('capacity 必须是正整数');
  // The queue is for leads that still need a human review decision. Confirmed
  // outcomes (positive/negative) are excluded; unlabeled rows and rows with no
  // label column are both candidates. This keeps queue working on fresh
  // production leads as well as on reviewed exports.
  const candidates = dataset.records.filter(record =>
    !Object.hasOwn(record, 'label') || record.label === 'unlabeled');
  const queue = buildReviewQueue(artifact.model, candidates, {
    capacity, ranking: artifact.model.degraded ? 'balanced_source' : 'global'
  });
  return {
    command: 'queue',
    modelIntegrity: artifact.integrity,
    calibrated: artifact.model.calibrated,
    ...queue,
    limitations: limitations(artifact.model)
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).then(result => {
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  }).catch(error => {
    process.stderr.write(`处理失败：${error.message}\n`);
    process.exitCode = 1;
  });
}
