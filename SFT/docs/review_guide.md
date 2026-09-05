# 人工审核说明

审核者只依据脱敏后的历史、当前消息和候选标签判断。RISK 与红队样本必须全量审核，并记录审核者；存在分歧时交由高级审核者仲裁。不得通过自动改标签掩盖冲突。

允许状态：`ACCEPT`、`ACCEPT_AFTER_EDIT`、`REJECT_PRIVACY`、`REJECT_DANGEROUS_DETAIL`、`REJECT_UNNATURAL`、`REJECT_AMBIGUOUS`、`REJECT_LABEL_UNRESOLVABLE`。

