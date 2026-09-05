from mindbridge_sft.classify import build_classification_prompt
from mindbridge_sft.models import ProblemCandidate


def test_classifier_prompt_excludes_generation_metadata():
    candidate = ProblemCandidate(
        id="x", text="我在论文中讨论自杀风险，不是在说我自己。",
        scene="论文", boundary_type="风险边界", ambiguity_note="应判普通任务",
    )
    prompt = build_classification_prompt(candidate)
    assert "论文\n" not in prompt
    assert "风险边界" not in prompt
    assert "应判普通任务" not in prompt
    assert candidate.text in prompt

