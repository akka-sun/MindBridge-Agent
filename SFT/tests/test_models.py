import pytest
from pydantic import ValidationError

from mindbridge_sft.models import JointLabel, ProblemCandidate


def test_joint_label_rejects_unknown_value():
    with pytest.raises(ValidationError):
        JointLabel(intent="OTHER", emotion="NORMAL", risk="LOW")


def test_candidate_requires_nonempty_text():
    with pytest.raises(ValidationError):
        ProblemCandidate(id="x", text="   ")

