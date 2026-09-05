from __future__ import annotations


HIGH_RISK_WORDS = ("想自杀","不想活","伤害自己","自残","结束生命","控制不住","伤害别人")
ACADEMIC_WORDS = ("论文","作业","翻译","定义","新闻","政策","术语","研究")
CONSULT_WORDS = ("焦虑","紧张","压力","睡不着","失眠","低落","孤独","崩溃","没动力","绝望")
DEPRESSED_WORDS = ("低落","孤独","崩溃","没动力","绝望","提不起兴趣")


def classify_by_rules(text: str) -> dict[str,str]:
    lowered=text.lower(); academic=any(x in lowered for x in ACADEMIC_WORDS)
    if not academic and any(x in lowered for x in HIGH_RISK_WORDS): return {"intent":"RISK","emotion":"HIGH_RISK","risk":"HIGH"}
    if any(x in lowered for x in CONSULT_WORDS):
        emotion="DEPRESSED" if any(x in lowered for x in DEPRESSED_WORDS) else "ANXIETY"
        risk="MEDIUM" if any(x in lowered for x in ("连续","很多天","两周","无法正常","绝望")) else "LOW"
        return {"intent":"CONSULT","emotion":emotion,"risk":risk}
    return {"intent":"CHAT","emotion":"NORMAL","risk":"LOW"}

