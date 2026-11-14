import { main } from '../src/cli.js';
import { readFile } from 'node:fs/promises';

const started = performance.now();
console.log('读取本地演练记录并核对输入指纹');
console.log(JSON.stringify(await main(['inspect']), null, 2));
console.log('现有领域核算');
console.log(JSON.stringify(await main(['summary','--out','var/summary.json']), null, 2));
console.log('业务配置');
console.log(JSON.stringify(JSON.parse(await readFile('data/policy.json', 'utf8')), null, 2));
console.log('PU 训练：区分 positive / negative / unlabeled，审阅样本只用于先验与验证');
const report = await main(['train','--out','var/artifact.json','--report','var/report.json']);
console.log(JSON.stringify({
  calibrated: report.calibrated,
  degraded: report.degraded,
  warnings: report.diagnostics.warnings,
  prior: report.priorSelection,
  evaluation: report.evaluation,
  reviewQueue: report.queue.queue
}, null, 2));
console.log('人工回访队列');
console.log(JSON.stringify(await main(['queue','--model','var/artifact.json','--capacity','5']), null, 2));
console.log('限制说明');
console.log(report.limitations.join('\n'));
console.log(`完成，用时 ${Math.ceil(performance.now() - started)} 毫秒`);
