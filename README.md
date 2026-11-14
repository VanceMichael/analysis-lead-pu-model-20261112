# lead-pu-model

label 取 positive/negative/unlabeled；reviewProbability 为独立审阅抽样概率，零表示未审阅。

项目提供本地记录校验、输入指纹、按 `label` 的原始核算，以及一条 **PU（正例-未标注）风险学习** 流程，用来安排人工回访。
`unlabeled` 是“尚未回访”，**不是失败**；`negative` 是已完成回访后确认的失败。训练流程明确区分
`positive` / `negative` / `unlabeled`，不会把未标注记录当成负例。分析人员可以保留完整输入并复查每次导出的 JSON；
原始数据不被命令改写。业务配置位于 `data/policy.json`，字段与口径在 `docs/data.md`。

## 运行

使用 Node.js 22 LTS（>=22.22.1 且 <23）和 npm，无第三方运行依赖。
`.nvmrc` 选择 22 主版本，清单与锁文件记录相同的兼容范围；Dockerfile 仍固定基础镜像摘要，容器构建可精确复现。

```sh
npm ci --ignore-scripts --no-audit --no-fund --offline
npm run build
npm test
npm run summary
node src/cli.js inspect --input data/records.json
node src/cli.js summary --input data/records.json --out var/summary.json

# PU 训练：写模型与报告
node src/cli.js train --out var/artifact.json --report var/report.json
# 预测：按模型词典逐行评分
node src/cli.js predict --model var/artifact.json --input data/records.json --out var/predictions.json
# 人工回访队列：按企业上限输出
node src/cli.js queue --model var/artifact.json --capacity 10 --out var/queue.json

npm run demo
```

`npm run demo` 在本地打印输入、现有核算、PU 训练报告、回访队列与限制说明，并生成
`var/summary.json`、`var/artifact.json`、`var/report.json`，不访问网络、不等待真实业务时间，通常几秒内完成。
输出目录可删除后重新运行，测试只使用临时目录。

## PU 模型口径

训练流程（`src/pu.js`、`src/audit.js`、`src/evaluate.js`、`src/features.js`）：

- **标签分离**：`positive` 作为已确认正例；`unlabeled` 是正例/负例的混合，只通过 PU 风险估计进入训练；
  `negative` 来自既有回访选择，**不参与拟合**，也不能代表总体负例。
- **学习方案**：辅助逻辑回归区分“已标注正例”与“未标注行”，再按外部标注倾向换算为风险；
  未标注行从不被当作负例（非负 PU 风险）。实现不依赖第三方库。
- **先验来源**：类先验 `π` 由独立审阅样本按 `1/probability` 加权估计（Horvitz-Thompson），
  或退回 `policy.priorRange` 中点；绝不从未标注行重新估计。
- **标注倾向 `e`**：`e(source) = 已标注正例率 / 审阅先验`。SC 假设（“正例在各来源层内随机被标注”）下，
  辅助分数 `g(x)` 估计 `e·P(y=1|x)`，因此校准风险 `r(x) = clip(g(x)/e, 0, 1)`；**不是**除以类先验。
- **假设检验**：若某来源 `e` 点估计 > 1、区间下界 > 1，或各来源 `e` 的极差超过 `scTolerance`，
  说明 SC 不成立；模型**降级为未校准排序**（`risk: null`），并输出按来源层的敏感性范围。
- **样本量门槛**：训练侧正例少于 `minPositives` 直接拒绝训练；少于 `minCalibrationPositives`
  时仍训练但降级为未校准排序。
- **先验越界**：审阅估计落在 `priorRange` 外时退回区间中点并标记 `prior_out_of_scope`，不把越界值当事实。

## 训练与预测约束

- **特征只取 `score_at` 前可见属性**：`label`、`reviewProbability`、`id`、`company`、`known_at` 禁止作为特征；
  `features` 是允许清单，未知字段直接报错。若上游在 `score_at` 之后回填属性，模型无法检测（见限制说明）。
- **企业隔离**：训练与测试按企业切分，同一企业不会跨两侧；审阅验证样本的企业与训练、测试均不相交。
  切分会尝试若干确定性盐值，避免正例被随机全部落到测试侧。
- **审阅样本只做验证/先验**：它按 `probability` 加权计算 precision、recall、AUC 与**回访容量内正例覆盖**，
  不混入普通训练行；普通 `unlabeled` 行不用于计算真负率。
- **一次保存**：`artifact.json` 同时包含模型系数、预处理字典（特征、类别词表、数值矩、设计布局）、
  假设配置、审阅样本版本与输入指纹，并带完整性校验；`predict`/`queue` 只读取该 artifact，不重新拟合。
- **一致拒绝**：`predict`/`queue` 遇到缺列或训练时未见的类别，统一由同一变换路径报错
  （`missingColumn` / `unknownCategory`），退出码非零。

## 容器

```sh
docker build -t lead-pu-model:local .
docker run --rm --network none lead-pu-model:local
```

Dockerfile 固定 Node.js 基础镜像摘要，构建时执行语法检查和测试。镜像构建完成后演示无需联网；可覆盖命令运行 `npm test`。构建不需要真实账号、邮件服务、数据库或线上数据。

## 目录

- `src/dataset.js`：记录校验、规范序列化和数据指纹。
- `src/domain.js`：现有业务分组核算。
- `src/features.js`：特征允许清单、可见性约束、预处理字典与变换。
- `src/stats.js`：确定性数值工具（逻辑回归、加权 AUC、Wilson 区间）。
- `src/audit.js`：审阅样本校验、逆概率先验、标注倾向与 SC 冲突检测。
- `src/pu.js`：PU 风险学习、企业切分、回访队列与限制说明。
- `src/evaluate.js`：审阅侧逆概率加权指标；真负率只对确认负例开放。
- `src/model.js`：artifact 组装、完整性校验与预测入口。
- `src/cli.js`：`summary` / `inspect` / `train` / `predict` / `queue` 入口。
- `src/output.js`：以临时文件写入后替换完整 JSON。
- `scripts/synthetic.js`：可控标签缺失机制的小数据生成器（用于测试与演示）。
- `test/`：输入约束、领域核对、PU 假设、命令与输出回归。
- `data/`：合成记录、独立核对值及本地业务配置。
