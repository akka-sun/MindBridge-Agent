# MindBridge 联合意图识别 SFT 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可复现的 DeepSeek v4 两阶段数据生成、人工审核、联合 QLoRA SFT 和 SFT 前后评测流水线。

**Architecture:** 数据先由 DeepSeek v4 分别生成普通问题和特殊边界问题，再由独立分类调用产生候选标签。候选数据经过 Schema、隐私、危险内容、重复和标签一致性检查，以及分层人工审核后，按 `split_group` 切分并用于联合分类 SFT。评测脚本对基础模型和 SFT 模型使用同一冻结测试集，分别计算 intent、emotion、risk 和安全指标。

**Tech Stack:** Python 3.11、PyTorch、Transformers、PEFT、TRL、Accelerate、Pydantic、jsonschema、scikit-learn、httpx、pytest。

**Spec:** `../意图与情绪联合SFT数据设计文档.md`

## Global Constraints

- 只使用 SFT/LoRA/QLoRA，不引入 GRPO、PPO 或其他强化学习。
- 第一版训练目标严格为 `intent`、`emotion`、`risk` 三字段 JSON，不训练自然语言解释。
- 训练 4,200 条、验证 900 条、测试 900 条、红队 600 条。
- 所有 RISK 和红队样本必须人工审核。
- 测试集和红队集不得参与 Prompt 调试、训练和 checkpoint 选择。
- 线上高风险硬规则优先，SFT 模型不能降低 HIGH 判定。
- 原始响应、Prompt 版本、模型名称、生成参数和审核记录必须可追踪。
- 所有真实日志在送入外部 API 前必须脱敏并确认数据使用授权。

---

## 文件结构与职责

| 文件 | 职责 |
|---|---|
| `configs/data_generation.yaml` | DeepSeek 生成批次、场景、参数和目标数量 |
| `configs/train_qlora.yaml` | 基座模型、LoRA 和训练参数 |
| `schemas/*.json` | 问题生成和联合标签 JSON Schema |
| `prompts/*.txt` | 版本化的三类 Prompt |
| `src/mindbridge_sft/models.py` | 数据模型和枚举 |
| `src/mindbridge_sft/deepseek_client.py` | API 调用、重试和原始响应落盘 |
| `src/mindbridge_sft/generate.py` | 普通/边界问题生成 CLI |
| `src/mindbridge_sft/classify.py` | 独立联合分类 CLI |
| `src/mindbridge_sft/validate.py` | Schema、隐私、危险内容、冲突和重复检查 |
| `src/mindbridge_sft/review_export.py` | 人工审核 CSV 导出和导入 |
| `src/mindbridge_sft/split.py` | 按组分层切分并生成 manifest/hash |
| `src/mindbridge_sft/train.py` | TRL + PEFT QLoRA 训练 |
| `src/mindbridge_sft/evaluate.py` | 基础/SFT 模型统一推理和指标计算 |
| `src/mindbridge_sft/compare.py` | 生成 SFT 前后 Markdown 对比报告 |
| `tests/` | 每个模块的单元和集成测试 |

---

### Task 1: 建立 Python 包、配置与数据模型

**Files:**
- Create: `SFT/pyproject.toml`
- Create: `SFT/configs/data_generation.yaml`
- Create: `SFT/configs/train_qlora.yaml`
- Create: `SFT/src/mindbridge_sft/__init__.py`
- Create: `SFT/src/mindbridge_sft/models.py`
- Test: `SFT/tests/test_models.py`

**Interfaces:**
- Produces: `Intent`, `Emotion`, `Risk`, `ProblemCandidate`, `JointLabel`, `ReviewedRecord`。

- [ ] **Step 1: 编写数据模型失败测试**

```python
def test_joint_label_rejects_unknown_value():
    with pytest.raises(ValidationError):
        JointLabel(intent="OTHER", emotion="NORMAL", risk="LOW")
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
pytest tests/test_models.py -q
```

预期：因 `mindbridge_sft.models` 尚不存在而失败。

- [ ] **Step 3: 实现枚举和 Pydantic 模型**

```python
class Intent(str, Enum):
    CHAT = "CHAT"
    CONSULT = "CONSULT"
    RISK = "RISK"

class JointLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Intent
    emotion: Emotion
    risk: Risk
```

- [ ] **Step 4: 增加配置并验证解析**

`data_generation.yaml` 必须包含普通场景、10 类边界场景、批大小、温度、重试次数和目标候选数；`train_qlora.yaml` 必须包含基座模型、LoRA、数据和输出配置。

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_models.py -q
```

预期：PASS。

### Task 2: 固化 Prompt 与 JSON Schema

**Files:**
- Create: `SFT/prompts/ordinary_problem.txt`
- Create: `SFT/prompts/boundary_problem.txt`
- Create: `SFT/prompts/classify_joint.txt`
- Create: `SFT/schemas/ordinary_problem.schema.json`
- Create: `SFT/schemas/boundary_problem.schema.json`
- Create: `SFT/schemas/joint_label.schema.json`
- Test: `SFT/tests/test_schemas.py`

**Interfaces:**
- Produces: 可由 `jsonschema.validate(instance, schema)` 验证的三套 Schema。
- Consumes: 设计文档第 5～7 节中的 Prompt 和 Schema。

- [ ] **Step 1: 编写 Schema 正反例测试**

```python
def test_joint_schema_rejects_extra_fields(joint_schema):
    value = {"intent": "CHAT", "emotion": "NORMAL", "risk": "LOW", "answer": "x"}
    with pytest.raises(ValidationError):
        validate(value, joint_schema)
```

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_schemas.py -q
```

- [ ] **Step 3: 将设计文档中的完整 Prompt 写入独立文件**

Prompt 文件首行加入版本注释，例如 `# prompt_version: ordinary-v1.0`。加载器忽略以 `#` 开头的元数据行。

- [ ] **Step 4: 写入 Schema，并设置 `additionalProperties: false`**

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_schemas.py -q
```

预期：所有合法样本通过，额外字段、非法枚举和缺字段样本失败。

### Task 3: 实现 DeepSeek v4 客户端

**Files:**
- Create: `SFT/src/mindbridge_sft/deepseek_client.py`
- Test: `SFT/tests/test_deepseek_client.py`

**Interfaces:**
- Produces: `DeepSeekClient.generate_json(system_prompt: str, user_prompt: str, schema: dict, temperature: float) -> dict | list[dict]`。
- Consumes: `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`。

- [ ] **Step 1: 编写超时、429、非法 JSON 和成功响应测试**

使用 `httpx.MockTransport`，不能在单元测试中调用真实 API。

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_deepseek_client.py -q
```

- [ ] **Step 3: 实现兼容 Chat Completions 的客户端**

客户端必须保存 request ID、模型、参数、Prompt hash、原始响应和解析错误；指数退避最多三次，401/403 不重试。

- [ ] **Step 4: 加入 Schema 校验**

响应 JSON 解析成功但 Schema 不合法时视为失败，不能静默删除字段。

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_deepseek_client.py -q
```

### Task 4: 实现普通与边界问题生成

**Files:**
- Create: `SFT/src/mindbridge_sft/generate.py`
- Test: `SFT/tests/test_generate.py`

**Interfaces:**
- Produces: `generate_candidates(kind: Literal["ordinary", "boundary"], config: GenerationConfig, output: Path) -> GenerationReport`。
- Consumes: Task 2 Prompt/Schema、Task 3 `DeepSeekClient`。

- [ ] **Step 1: 编写批次生成和断点续跑测试**

测试同一 batch ID 已成功时不会重复调用 API，失败批次进入 `failed_batches.jsonl`。

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_generate.py -q
```

- [ ] **Step 3: 实现 CLI 和 JSONL 原子追加**

输出记录必须包含 `id`、`text`、可选 `history`、`generation_batch_id`、`prompt_version`、`model` 和生成参数。标签不得在本阶段写入。

- [ ] **Step 4: 加入 ID 冲突检查**

- [ ] **Step 5: 运行测试并执行 10 条 mock 冒烟测试**

```bash
pytest tests/test_generate.py -q
python -m mindbridge_sft.generate --kind ordinary --config configs/data_generation.yaml --dry-run
```

### Task 5: 实现独立联合分类

**Files:**
- Create: `SFT/src/mindbridge_sft/classify.py`
- Test: `SFT/tests/test_classify.py`

**Interfaces:**
- Produces: `classify_candidates(inputs: list[Path], output: Path) -> ClassificationReport`。
- Consumes: Task 3 客户端、Task 4 候选；仅向模型传递 `history` 和 `text`。

- [ ] **Step 1: 编写防标签泄漏测试**

```python
def test_classifier_prompt_excludes_generation_metadata(candidate):
    prompt = build_classification_prompt(candidate)
    assert candidate.scene not in prompt
    assert candidate.boundary_type not in prompt
    assert candidate.ambiguity_note not in prompt
```

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_classify.py -q
```

- [ ] **Step 3: 实现分类、候选置信度和 reason_codes 落盘**

训练目标仍只包含三个标签；置信度和 reason codes 只用于质检。

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_classify.py -q
```

### Task 6: 实现自动数据质检

**Files:**
- Create: `SFT/src/mindbridge_sft/validate.py`
- Create: `SFT/src/mindbridge_sft/quality_rules.py`
- Test: `SFT/tests/test_validate.py`

**Interfaces:**
- Produces: `validate_records(records) -> tuple[list[Record], list[ReviewIssue], QualityReport]`。
- Consumes: 联合候选 JSONL。

- [ ] **Step 1: 编写隐私、危险细节、元话语、空标签和冲突组合测试**

- [ ] **Step 2: 编写完全重复、规范化重复和相似重复测试**

- [ ] **Step 3: 运行并确认失败**

```bash
pytest tests/test_validate.py -q
```

- [ ] **Step 4: 实现拒绝与复核分流**

危险细节和未脱敏隐私直接拒绝；标签冲突、边界和低置信度进入人工复核。程序不得自动改写金标准标签。

- [ ] **Step 5: 生成质量报告**

报告包含每种拒绝原因、复核原因、重复率、各标签分布和各生成批次错误率。

- [ ] **Step 6: 运行测试**

```bash
pytest tests/test_validate.py -q
```

### Task 7: 实现人工审核导入导出

**Files:**
- Create: `SFT/src/mindbridge_sft/review_export.py`
- Create: `SFT/docs/review_guide.md`
- Test: `SFT/tests/test_review_export.py`

**Interfaces:**
- Produces: `export_review_csv(...)`、`import_review_csv(...)`。
- Consumes: Task 6 的通过集和复核集。

- [ ] **Step 1: 编写 CSV 往返测试**

确保 UTF-8 BOM、中文内容、历史 JSON、审核状态、审核者和修正标签往返后不丢失。

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_review_export.py -q
```

- [ ] **Step 3: 实现抽样策略**

RISK 和红队 100%；CONSULT/CHAT 边界 30%；普通 CHAT/CONSULT 按场景、批次分层抽取 10%。批次抽查错误率超过 5% 时，将整批标为需要复核。

- [ ] **Step 4: 实现导入验证**

只接受 `ACCEPT`、`ACCEPT_AFTER_EDIT` 和设计文档列出的拒绝状态；RISK 必须具备审核记录。

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_review_export.py -q
```

### Task 8: 实现分组切分与数据清单

**Files:**
- Create: `SFT/src/mindbridge_sft/split.py`
- Test: `SFT/tests/test_split.py`

**Interfaces:**
- Produces: `train.jsonl`、`valid.jsonl`、`test.jsonl`、`redteam_test.jsonl`、`dataset_manifest.json`、`split_hashes.json`。
- Consumes: Task 7 审核通过数据。

- [ ] **Step 1: 编写 `split_group` 不交叉测试**

```python
assert train_groups.isdisjoint(valid_groups)
assert train_groups.isdisjoint(test_groups)
assert valid_groups.isdisjoint(test_groups)
```

- [ ] **Step 2: 编写类别数量和 hash 测试**

- [ ] **Step 3: 运行并确认失败**

```bash
pytest tests/test_split.py -q
```

- [ ] **Step 4: 实现固定种子分组分层切分**

目标数量严格为 4,200/900/900/600。若审核后的某一类别不足，脚本必须失败并报告缺口，不能复制样本补齐。

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_split.py -q
```

### Task 9: 实现统一评测器和规则基线

**Files:**
- Create: `SFT/src/mindbridge_sft/evaluate.py`
- Create: `SFT/src/mindbridge_sft/metrics.py`
- Create: `SFT/src/mindbridge_sft/rule_baseline.py`
- Test: `SFT/tests/test_metrics.py`
- Test: `SFT/tests/test_rule_baseline.py`

**Interfaces:**
- Produces: `evaluate_predictions(gold, predictions) -> EvaluationReport`。
- Consumes: `test.jsonl`、`redteam_test.jsonl`、基础或 SFT 模型。

- [ ] **Step 1: 手工构造小型混淆矩阵测试**

断言 Accuracy、Macro-F1、RISK Recall、HIGH FNR 和三字段完全匹配率的精确值。

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_metrics.py tests/test_rule_baseline.py -q
```

- [ ] **Step 3: 实现规则-only 基线**

复用当前项目中的高风险词和咨询词逻辑，但复制到独立评测模块并固定版本，避免业务代码变化导致历史基线漂移。

- [ ] **Step 4: 实现模型推理和 JSON 解析**

保存每条原始输出、解析状态、预测、金标准、场景、延迟和错误类型。

- [ ] **Step 5: 输出总体、字段、组合、安全和分层指标**

- [ ] **Step 6: 运行测试**

```bash
pytest tests/test_metrics.py tests/test_rule_baseline.py -q
```

### Task 10: 实现 QLoRA SFT

**Files:**
- Create: `SFT/src/mindbridge_sft/train.py`
- Create: `SFT/src/mindbridge_sft/data_collator.py`
- Test: `SFT/tests/test_training_format.py`

**Interfaces:**
- Produces: LoRA adapter、trainer state、最佳 checkpoint、训练 manifest。
- Consumes: Task 8 train/valid，Task 1 训练配置。

- [ ] **Step 1: 编写 assistant-only loss mask 测试**

测试 system/user token 的 label 为 `-100`，assistant JSON token 参与 loss。

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_training_format.py -q
```

- [ ] **Step 3: 实现 tokenizer 模板与 QLoRA 加载**

训练只接受三字段固定顺序 JSON。训练启动时记录基座 revision、数据 hash、随机种子、包版本和全部超参数。

- [ ] **Step 4: 实现基于验证集的最佳 checkpoint 选择**

主要指标使用 Intent Macro-F1，并将 RISK Recall 作为不可下降约束；不能仅按 training loss 选 checkpoint。

- [ ] **Step 5: 运行小模型/少样本冒烟训练**

```bash
accelerate launch -m mindbridge_sft.train --config configs/train_qlora.yaml --train tests/fixtures/train.jsonl --valid tests/fixtures/valid.jsonl --max-steps 2 --output outputs/smoke
```

预期：完成两步、保存 adapter 和 manifest、无 NaN。

### Task 11: 实现 SFT 前后对比报告

**Files:**
- Create: `SFT/src/mindbridge_sft/compare.py`
- Test: `SFT/tests/test_compare.py`

**Interfaces:**
- Produces: `base_vs_sft_joint_eval.md` 和退化案例 JSONL。
- Consumes: 两份 `EvaluationReport`。

- [ ] **Step 1: 编写提升和退化识别测试**

SFT 的 HIGH Recall 下降时，报告必须显示阻断上线；总体 Accuracy 上升不能覆盖安全退化。

- [ ] **Step 2: 运行并确认失败**

```bash
pytest tests/test_compare.py -q
```

- [ ] **Step 3: 实现 Markdown 报告**

报告包含指标差值、混淆矩阵、分层场景、JSON 失败、最大退化类别和失败案例链接。

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_compare.py -q
```

### Task 12: 接入 MindBridge 运行时

**Files:**
- Create: `mindbridge-py 0612/mindbridge-py 3/app/services/joint_classifier.py`
- Modify: `mindbridge-py 0612/mindbridge-py 3/app/agents/runtime.py`
- Modify: `mindbridge-py 0612/mindbridge-py 3/app/services/assessment.py`
- Modify: `mindbridge-py 0612/mindbridge-py 3/app/core/config.py`
- Test: `mindbridge-py 0612/mindbridge-py 3/tests/test_joint_classifier.py`
- Test: `mindbridge-py 0612/mindbridge-py 3/tests/test_chat_pipeline.py`

**Interfaces:**
- Produces: `JointClassifier.classify(text: str, history: list[AiMessage]) -> JointPrediction`。
- Consumes: SFT 后的模型或 OpenAI-compatible 本地推理接口。

- [ ] **Step 1: 编写高风险不可降级测试**

```python
def test_hard_rule_high_cannot_be_downgraded():
    model = FakeClassifier(intent="CONSULT", emotion="ANXIETY", risk="LOW")
    result = service.classify("我现在想伤害自己", [])
    assert result.intent == IntentType.RISK
    assert result.risk == RiskLevel.HIGH
```

- [ ] **Step 2: 编写模型超时和非法 JSON 降级测试**

- [ ] **Step 3: 运行并确认失败**

```bash
pytest tests/test_joint_classifier.py tests/test_chat_pipeline.py -q
```

- [ ] **Step 4: 实现联合分类服务和配置开关**

保留旧分类路径作为可回滚开关；先 shadow 记录新旧结果，不立即改变告警行为。

- [ ] **Step 5: 运行完整测试和离线评测**

```bash
pytest -q
python -m mindbridge_sft.compare --before reports/base_joint_eval.json --after reports/sft_joint_eval.json --output reports/base_vs_sft_joint_eval.md
```

- [ ] **Step 6: 人工审核全部高风险差异案例后再灰度**

---

## 完成定义

只有以下条件全部满足才算实施完成：

- 三类 Prompt 和三套 Schema 已版本化；
- 两阶段 DeepSeek v4 数据流水线可断点续跑；
- 自动质检不会静默修正标签；
- 人工审核有抽样、双审、仲裁和审计记录；
- 数据按组切分且无泄漏；
- 基础模型、规则-only、SFT 裸模型和规则合并模型均有报告；
- HIGH 风险安全指标没有退化；
- 训练配置、数据 hash、模型 revision 和依赖版本可以复现；
- MindBridge 运行时支持 shadow、灰度和回滚。

