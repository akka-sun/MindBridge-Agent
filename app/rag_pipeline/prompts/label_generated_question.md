---
prompt_id: label_generated_question
version: 1.0.0
model_family: deepseek
schema: ../schemas/labeled_question.schema.json
temperature: 0.1
max_output_tokens: 1000
---
你是独立的评测标注员。根据候选问题和真实章节证据判断是否可回答。
普通示例：问题事实能在quote中逐字找到时accept，并绑定section_id。
边界示例：问题要求中国热线但证据只有美国988时reject，gold_evidence为空并说明地区信息缺失。
失败示例：问题前提错误、超出资料或要求伤害方法时reject，不要替问题改写答案。
每条必要事实都必须绑定可逐字定位的quote；不可凭常识补充。

候选问题：$question
候选证据：$evidence
