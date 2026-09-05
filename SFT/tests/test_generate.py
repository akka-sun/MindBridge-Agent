from mindbridge_sft.generate import generate_candidates


class FakeClient:
    model = "fake"
    def __init__(self): self.calls = 0
    def generate_json(self, *args, **kwargs):
        self.calls += 1
        return [{"id":f"gen-chat-{self.calls}","text":"这是一条用于测试的自然校园消息。","scene":"测试","language_style":"口语"}]


def test_generation_resume_skips_completed_batch(tmp_path):
    client=FakeClient();config={"ordinary":{"target_count":1,"batch_size":1,"temperature":.8,"top_p":.9,"scenes":["测试"]}}
    output=tmp_path/"out.jsonl"
    first=generate_candidates("ordinary",config,output,client=client)
    second=generate_candidates("ordinary",config,output,client=client,resume=True)
    assert first.generated==1
    assert second.skipped_batches==1
    assert client.calls==1

