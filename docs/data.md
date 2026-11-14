# 数据与口径

label 取 positive/negative/unlabeled；reviewProbability 为独立审阅抽样概率，零表示未审阅。

`data/records.json` 是版本为 1 的演练数据，所有人员、机构、商品和交易均为本地合成记录。`id` 是导入记录键，不自动代表企业、访问或其他统计单位；记录键不得重复。数值字段必须有限，整数字段不得超过安全整数范围。空字符串可用于字段约定允许的未公开标签，不用零冒充缺测。

| 字段 | 类型 |
| --- | --- |
| `id` | `string` |
| `company` | `string` |
| `score_at` | `integer` |
| `known_at` | `integer` |
| `source` | `string` |
| `size` | `number` |
| `engagement` | `number` |
| `label` | `string` |
| `reviewProbability` | `number` |

## 标签语义（PU 训练用）

| label | 含义 | 在训练中的角色 |
| --- | --- | --- |
| `positive` | 已确认成交/正例 | 已标注正例 |
| `negative` | 已完成回访并确认失败 | **不参与拟合**；仅作偏倚诊断 |
| `unlabeled` | 尚未回访 | 正例/负例混合；只经 PU 风险估计进入训练 |

销售把“尚未回访”的商机记成失败会引入选择偏倚：模型只会追着已被销售偏爱的大客户。因此本流程**不把
`unlabeled` 当负例，也不把 `negative` 当作总体负例**。普通 `unlabeled` 行不用于计算真负率；
真负率只在确认负例上定义，且注明其选择偏倚。

## 审阅样本（`policy.reviewSample`）

独立审阅的小样本，含 `company / size / engagement / source / positive / probability / score_at / known_at`。
`probability` 是该行的独立抽样概率（`0 < probability <= 1`），用于逆概率加权。审阅样本：

- **只用于**估计类先验与按来源的标注倾向，以及在审阅侧做加权验证；
- **不并入**普通训练行；
- 其企业必须与训练、测试企业均不相交，否则训练报 `companyLeakage`。

## PU 口径

- 类先验 `π`：审阅样本的 `1/probability` 加权正例率；越出 `policy.priorRange` 时退回区间中点并标记。
- 标注倾向 `e(source) = 已标注正例率 / π(source)`。SC（“正例在各来源层内随机被标注”）成立时
  校准风险 `r(x) = clip(g(x)/e, 0, 1)`；不成立时降级为未校准排序，`risk` 为 `null`。
- 敏感性范围：按来源分别给出 `e` 及其区间；`artifact.sensitivity.perSource` 即该范围。
- 分数**不是**成交概率；即使校准，也只是假设下的正例风险。

## 现有命令口径

`summary` 按 `label` 清点记录，`sum` 为 `engagement` 的逐行核对值，不代表统计推断或跨期业务指标。分组按键排序，行按记录键稳定处理。`data/expected-summary.json` 是独立生成的现有核算对照。

`inspect` 返回输入行数、字段列表与 SHA-256 指纹；指纹不依赖字段或记录排列。数据版本不支持时应直接拒绝，不做静默迁移。错误发生时命令返回非零退出码。

## 配置字段（`data/policy.json`）

| 字段 | 作用 |
| --- | --- |
| `features` | 允许作为特征的字段清单；禁止 `label/reviewProbability/id/company/known_at` |
| `capacity` | 回访容量（按企业计） |
| `priorRange` | 业务可接受的先验范围；越界先验不采用 |
| `scMechanism` | 标注机制说明（默认 `bernoulli`） |
| `scTolerance` | 各来源标注倾向可接受的极差 |
| `minPositives` | 训练侧最少确认正例数；不足则拒绝训练 |
| `minCalibrationPositives` | 允许校准的最少正例数；不足则降级为未校准排序 |
| `l2` | 辅助逻辑回归的 L2 正则 |
| `reviewSample` | 独立审阅样本 |
