# MindBridge 联合分类 SFT：Linux 运行指南

本文所有命令仅适用于 Linux。请在 `SFT` 目录下执行命令。

> 下面的 `mindbridge_sft` 命令是本目录实施方案约定的统一运行入口；完成对应脚本实现后，可按本文顺序直接运行。

## 1. 环境要求

- Ubuntu 22.04 或 24.04
- Python 3.11
- NVIDIA GPU 与可用的 CUDA 驱动
- 建议显存：24 GB 及以上；显存较小时使用 QLoRA 并减小批大小
- 建议可用磁盘空间：50 GB 及以上
- DeepSeek API Key

确认 GPU 和驱动可用：

```bash
nvidia-smi
```

## 2. 创建 Python 环境

```bash
cd /path/to/mindbridge/SFT

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

## 3. 安装依赖

先根据服务器 CUDA 版本安装 PyTorch。以下命令以 CUDA 12.4 为例：

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

安装项目依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

验证 PyTorch 和 CUDA：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 4. 配置环境变量

```bash
cp .env.example .env
chmod 600 .env
```

编辑 `.env`：

```bash
nano .env
```

至少配置以下项目：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
BASE_MODEL_PATH=/path/to/base-model
```

加载环境变量：

```bash
set -a
source .env
set +a
```

## 5. 创建运行目录

```bash
mkdir -p \
  data/raw \
  data/classified \
  data/validated \
  data/review \
  data/splits \
  reports \
  outputs
```

## 6. 运行测试

```bash
pytest -q
```

## 7. 检查项目已有参考数据

```bash
python -m mindbridge_sft.validate_reference \
  --input "../mindbridge-py 0612/data/psychqa_synthetic.jsonl" \
  --report reports/reference_data_report.json
```

## 8. 小批量验证 DeepSeek 数据生成

先生成少量普通问题：

```bash
python -m mindbridge_sft.generate_questions \
  --config configs/data_generation.yaml \
  --question-type ordinary \
  --count 20 \
  --output data/raw/ordinary_smoke.jsonl
```

再生成少量特殊边界问题：

```bash
python -m mindbridge_sft.generate_questions \
  --config configs/data_generation.yaml \
  --question-type boundary \
  --count 20 \
  --output data/raw/boundary_smoke.jsonl
```

验证生成结果：

```bash
python -m mindbridge_sft.validate_data \
  --inputs data/raw/ordinary_smoke.jsonl \
  --schema schemas/ordinary_problem.schema.json \
  --report reports/ordinary_smoke_report.json

python -m mindbridge_sft.validate_data \
  --inputs data/raw/boundary_smoke.jsonl \
  --schema schemas/boundary_problem.schema.json \
  --report reports/boundary_smoke_report.json
```

## 9. 批量生成普通问题和边界问题

生成普通问题：

```bash
python -m mindbridge_sft.generate_questions \
  --config configs/data_generation.yaml \
  --question-type ordinary \
  --count 4800 \
  --output data/raw/ordinary.jsonl \
  --resume
```

生成特殊边界问题：

```bash
python -m mindbridge_sft.generate_questions \
  --config configs/data_generation.yaml \
  --question-type boundary \
  --count 1800 \
  --output data/raw/boundary.jsonl \
  --resume
```

## 10. 对问题进行意图与情绪分类

分类普通问题：

```bash
python -m mindbridge_sft.classify_questions \
  --config configs/classification.yaml \
  --input data/raw/ordinary.jsonl \
  --output data/classified/ordinary.jsonl \
  --resume
```

分类边界问题：

```bash
python -m mindbridge_sft.classify_questions \
  --config configs/classification.yaml \
  --input data/raw/boundary.jsonl \
  --output data/classified/boundary.jsonl \
  --resume
```

## 11. 自动校验和去重

```bash
python -m mindbridge_sft.validate_data \
  --inputs data/classified/ordinary.jsonl data/classified/boundary.jsonl \
  --deduplicate \
  --output data/validated/all.jsonl \
  --rejected-output data/validated/rejected.jsonl \
  --report reports/validation_report.json
```

## 12. 人工抽查

导出待审核样本：

```bash
python -m mindbridge_sft.sample_for_review \
  --input data/validated/all.jsonl \
  --output data/review/review_samples.csv \
  --ordinary-count 300 \
  --boundary-count 300 \
  --seed 42
```

人工审核并保存 CSV 后，导入审核结果：

```bash
python -m mindbridge_sft.import_review \
  --data data/validated/all.jsonl \
  --review data/review/review_samples.csv \
  --output data/validated/approved.jsonl \
  --rejected-output data/validated/manual_rejected.jsonl \
  --report reports/manual_review_report.json
```

## 13. 划分训练集、验证集和测试集

```bash
python -m mindbridge_sft.split_data \
  --input data/validated/approved.jsonl \
  --output-dir data/splits \
  --train-size 4200 \
  --valid-size 900 \
  --test-size 900 \
  --redteam-size 600 \
  --stratify-by intent emotion question_type \
  --group-by semantic_group_id \
  --seed 42 \
  --report reports/split_report.json
```

检查切分结果：

```bash
python -m mindbridge_sft.validate_splits \
  --data-dir data/splits \
  --report reports/split_validation_report.json
```

## 14. 测评微调前模型

```bash
python -m mindbridge_sft.evaluate \
  --model "$BASE_MODEL_PATH" \
  --test-file data/splits/test.jsonl \
  --redteam-file data/splits/redteam.jsonl \
  --output reports/baseline_metrics.json \
  --predictions reports/baseline_predictions.jsonl
```

## 15. 配置 Accelerate

```bash
accelerate config
accelerate env
```

单卡运行时，也可以直接使用默认配置：

```bash
accelerate config default
```

## 16. 冒烟训练

```bash
accelerate launch -m mindbridge_sft.train \
  --config configs/train_qlora.yaml \
  --train-file data/splits/train.jsonl \
  --valid-file data/splits/valid.jsonl \
  --output-dir outputs/smoke \
  --max-steps 20 \
  --logging-steps 1 \
  --save-steps 20
```

## 17. 正式进行 QLoRA SFT

```bash
accelerate launch -m mindbridge_sft.train \
  --config configs/train_qlora.yaml \
  --train-file data/splits/train.jsonl \
  --valid-file data/splits/valid.jsonl \
  --output-dir outputs/intent_emotion_qlora
```

如需从检查点继续：

```bash
accelerate launch -m mindbridge_sft.train \
  --config configs/train_qlora.yaml \
  --train-file data/splits/train.jsonl \
  --valid-file data/splits/valid.jsonl \
  --output-dir outputs/intent_emotion_qlora \
  --resume-from-checkpoint outputs/intent_emotion_qlora/checkpoint-last
```

## 18. 测评微调后模型

```bash
python -m mindbridge_sft.evaluate \
  --model "$BASE_MODEL_PATH" \
  --adapter outputs/intent_emotion_qlora \
  --test-file data/splits/test.jsonl \
  --redteam-file data/splits/redteam.jsonl \
  --output reports/sft_metrics.json \
  --predictions reports/sft_predictions.jsonl
```

## 19. 生成微调前后对比报告

```bash
python -m mindbridge_sft.compare_metrics \
  --baseline reports/baseline_metrics.json \
  --sft reports/sft_metrics.json \
  --output reports/sft_comparison.md
```

查看报告：

```bash
sed -n '1,240p' reports/sft_comparison.md
```

## 20. 常见问题检查

CUDA 不可用：

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
```

训练显存不足时，在 `configs/train_qlora.yaml` 中减小：

```yaml
per_device_train_batch_size: 1
max_seq_length: 1024
```

并增大梯度累积：

```yaml
gradient_accumulation_steps: 16
```

检查 DeepSeek API 环境变量：

```bash
test -n "$DEEPSEEK_API_KEY" && echo "DEEPSEEK_API_KEY is set"
printf '%s\n' "$DEEPSEEK_BASE_URL"
printf '%s\n' "$DEEPSEEK_MODEL"
```

检查 JSONL 是否存在非法行：

```bash
python -m mindbridge_sft.validate_data \
  --inputs data/validated/all.jsonl \
  --report reports/jsonl_check.json
```

若通过审核的数据不足 6600 条，应先补充生成并重新执行分类、校验和审核，不要重复已有样本凑数。

若微调后的边界测试集指标下降，先检查 `reports/sft_predictions.jsonl` 中的误判样本，再补充同类型数据并重新训练。
