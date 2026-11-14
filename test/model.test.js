import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { loadDataset } from '../src/dataset.js';
import { fitPuModel } from '../src/pu.js';
import { buildArtifact, validateArtifact, predictRow, assertCompatible } from '../src/model.js';
import { validateAuditSample, auditVersion } from '../src/audit.js';
import { generateScenario, generateAuditSample } from '../scripts/synthetic.js';

const SIGNAL = { sizeWeight: 1.5, engagementWeight: 0 };
const POLICY = JSON.parse(await readFile('data/policy.json', 'utf8'));
const DATA = await loadDataset('data/records.json');

function artifactWith(records, policy, overrides = {}) {
  const fitted = fitPuModel(records, policy, {});
  const auditRows = validateAuditSample(policy.reviewSample ?? []);
  const artifact = buildArtifact(fitted.model, {
    policy,
    datasetFingerprint: 'test-fingerprint',
    reviewVersion: auditVersion(auditRows, policy),
    fittedAt: 'fixed'
  });
  return { artifact, fitted, ...overrides };
}

function syntheticPolicy() {
  const audit = generateAuditSample({ n: 2000, seed: 3, probability: 0.2,
    prevalence: { expo: 0.3, web: 0.3 }, signal: SIGNAL });
  return { features: ['size', 'engagement', 'source'], capacity: 10, priorRange: [0.1, 0.6],
    minPositives: 2, reviewSample: audit };
}

test('artifact 同时保存模型、预处理字典、假设配置与审阅样本版本', () => {
  const { records } = generateScenario({ n: 600, seed: 5, prevalence: { expo: 0.3, web: 0.3 },
    labelPropensity: { expo: 0.85, web: 0.85 }, negativeReview: 0.2, signal: SIGNAL });
  const policy = syntheticPolicy();
  const { artifact } = artifactWith(records, policy);
  assert.equal(artifact.artifactVersion, 1);
  assert.ok(artifact.model.coefficients.length > 0);
  assert.ok(Array.isArray(artifact.preprocessing.features));
  assert.ok(artifact.preprocessing.categories.source.includes('expo'));
  assert.ok(artifact.assumptions.scMechanism);
  assert.ok(artifact.assumptions.priorRange);
  assert.ok(artifact.reviewSampleVersion.length > 0);
  assert.ok(artifact.integrity.length > 0);
  validateArtifact(artifact);
});

test('artifact 完整性校验可发现篡改', () => {
  const { records } = generateScenario({ n: 600, seed: 5, prevalence: { expo: 0.3, web: 0.3 },
    labelPropensity: { expo: 0.85, web: 0.85 }, negativeReview: 0.2, signal: SIGNAL });
  const { artifact } = artifactWith(records, syntheticPolicy());
  const tampered = { ...artifact, model: { ...artifact.model, coefficients: [99] } };
  assert.throws(() => validateArtifact(tampered),
    error => error.code === 'artifactIntegrity');
});

test('审阅样本版本变化会改变 artifact 指纹', () => {
  const { records } = generateScenario({ n: 600, seed: 5, prevalence: { expo: 0.3, web: 0.3 },
    labelPropensity: { expo: 0.85, web: 0.85 }, negativeReview: 0.2, signal: SIGNAL });
  const policyA = syntheticPolicy();
  const policyB = { ...policyA, reviewSample: [...policyA.reviewSample, {
    company: 'auditextra', size: 12, engagement: 1, source: 'expo', positive: true,
    probability: 0.2, score_at: 40, known_at: 40 }] };
  const first = artifactWith(records, policyA).artifact;
  const second = artifactWith(records, policyB).artifact;
  assert.notEqual(first.reviewSampleVersion, second.reviewSampleVersion);
});

test('predictRow 与 queue 对缺列/未见类别返回同一错误码', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'pred-'));
  try {
    const { records } = generateScenario({ n: 600, seed: 5, prevalence: { expo: 0.3, web: 0.3 },
      labelPropensity: { expo: 0.85, web: 0.85 }, negativeReview: 0.2, signal: SIGNAL });
    const policy = syntheticPolicy();
    const { artifact } = artifactWith(records, policy);
    const clean = records[0];
    const missing = { ...clean };
    delete missing.size;
    assert.throws(() => predictRow(artifact, missing), error => error.code === 'missingColumn');
    assert.throws(() => predictRow(artifact, { ...clean, source: 'brand-new' }),
      error => error.code === 'unknownCategory');

    const datasetPath = join(folder, 'data.json');
    await writeFile(datasetPath, JSON.stringify({ version: 1, records: [missing] }));
    const artifactPath = join(folder, 'model.json');
    await writeFile(artifactPath, JSON.stringify(artifact));
    const result = spawnSync(process.execPath,
      ['src/cli.js', 'predict', '--model', artifactPath, '--input', datasetPath], { encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /缺少字段/);

    await writeFile(datasetPath, JSON.stringify({ version: 1, records: [{ ...clean, source: 'brand-new' }] }));
    const unknown = spawnSync(process.execPath,
      ['src/cli.js', 'queue', '--model', artifactPath, '--input', datasetPath], { encoding: 'utf8' });
    assert.equal(unknown.status, 1);
    assert.match(unknown.stderr, /未出现在训练词典/);
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('predict 与 queue 命令行端到端可复核', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'e2e-'));
  try {
    const artifactPath = join(folder, 'artifact.json');
    const queuePath = join(folder, 'queue.json');
    const train = spawnSync(process.execPath,
      ['src/cli.js', 'train', '--out', artifactPath, '--report', join(folder, 'report.json')],
      { encoding: 'utf8' });
    assert.equal(train.status, 0, train.stderr);
    const artifact = JSON.parse(await readFile(artifactPath, 'utf8'));
    assert.ok(artifact.model.coefficients.length > 0);

    const queueRun = spawnSync(process.execPath,
      ['src/cli.js', 'queue', '--model', artifactPath, '--capacity', '4', '--out', queuePath],
      { encoding: 'utf8' });
    assert.equal(queueRun.status, 0, queueRun.stderr);
    const queue = JSON.parse(await readFile(queuePath, 'utf8'));
    assert.equal(queue.queue.length, 4);
    assert.equal(new Set(queue.queue.map(item => item.company)).size, 4);
    assert.ok(queue.limitations.length > 0);
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('演练数据上模型检测来源标注冲突并降级', async () => {
  const train = spawnSync(process.execPath, ['src/cli.js', 'train'], { encoding: 'utf8' });
  assert.equal(train.status, 0, train.stderr);
  const report = JSON.parse(train.stdout);
  assert.equal(report.calibrated, false);
  assert.equal(report.degraded, true);
  assert.ok(report.diagnostics.warnings.includes('sc_assumption_violated'));
  assert.ok(report.diagnostics.warnings.includes('prior_source_conflict'));
  assert.equal(report.diagnostics.evaluation.trueNegativeRate, null);
  assert.equal(report.artifacts === undefined, true);
});

test('assertCompatible 拒绝缺少模型字段的数据', () => {
  const { records } = generateScenario({ n: 600, seed: 5, prevalence: { expo: 0.3, web: 0.3 },
    labelPropensity: { expo: 0.85, web: 0.85 }, negativeReview: 0.2, signal: SIGNAL });
  const { artifact } = artifactWith(records, syntheticPolicy());
  const row = { ...records[0] };
  delete row.engagement;
  assert.throws(() => assertCompatible(artifact, [row]),
    error => error.code === 'missingColumn');
});

test('queue 可处理没有 label/company/id 的新商机', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'fresh-'));
  try {
    const datasetPath = join(folder, 'data.json');
    const artifactPath = join(folder, 'model.json');
    const train = spawnSync(process.execPath,
      ['src/cli.js', 'train', '--out', artifactPath], { encoding: 'utf8' });
    assert.equal(train.status, 0, train.stderr);
    const fresh = DATA.records.slice(0, 6).map(record => ({
      score_at: record.score_at, known_at: record.known_at, source: record.source,
      size: record.size, engagement: record.engagement
    }));
    await writeFile(datasetPath, JSON.stringify({ version: 1, records: fresh }));
    const run = spawnSync(process.execPath,
      ['src/cli.js', 'queue', '--model', artifactPath, '--input', datasetPath, '--capacity', '4'],
      { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr);
    const queue = JSON.parse(run.stdout);
    assert.equal(queue.queue.length, 4);
    assert.equal(queue.candidates, 6);
    for (const item of queue.queue) {
      assert.equal(item.company, null);
      assert.equal(item.id, null);
    }
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('predict 的输入不要求训练 schema 的 label/reviewProbability', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'lean-'));
  try {
    const artifactPath = join(folder, 'model.json');
    const datasetPath = join(folder, 'data.json');
    const train = spawnSync(process.execPath,
      ['src/cli.js', 'train', '--out', artifactPath], { encoding: 'utf8' });
    assert.equal(train.status, 0, train.stderr);
    const record = DATA.records[0];
    const lean = { id: record.id, company: record.company, score_at: record.score_at,
      source: record.source, size: record.size, engagement: record.engagement };
    await writeFile(datasetPath, JSON.stringify({ version: 1, records: [lean] }));
    const run = spawnSync(process.execPath,
      ['src/cli.js', 'predict', '--model', artifactPath, '--input', datasetPath],
      { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr);
    const result = JSON.parse(run.stdout);
    assert.equal(result.predictions.length, 1);
    assert.equal(typeof result.predictions[0].score, 'number');
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('真实数据指纹进入 artifact', () => {
  const fitted = fitPuModel(DATA.records, POLICY, { datasetFingerprint: DATA.fingerprint });
  const artifact = buildArtifact(fitted.model, {
    policy: POLICY, datasetFingerprint: DATA.fingerprint,
    reviewVersion: auditVersion(validateAuditSample(POLICY.reviewSample), POLICY), fittedAt: null
  });
  assert.equal(artifact.datasetFingerprint, DATA.fingerprint);
  assert.equal(fitted.model.datasetFingerprint, DATA.fingerprint);
  // The real data is the deliberate confound: source fully determines the
  // observed label, so the audit boundary must be declared to the artifact.
  assert.ok(fitted.model.degraded);
});
