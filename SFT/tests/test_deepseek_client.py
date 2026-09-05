import json

import httpx
import pytest

from mindbridge_sft.deepseek_client import DeepSeekClient, DeepSeekError


SCHEMA = {"type":"object","additionalProperties":False,"required":["ok"],"properties":{"ok":{"type":"boolean"}}}


def test_successful_json_response_is_validated(tmp_path):
    def handler(request):
        return httpx.Response(200,json={"choices":[{"message":{"content":"{\"ok\":true}"}}]})
    client=DeepSeekClient(api_key="x",transport=httpx.MockTransport(handler),audit_path=tmp_path/"audit.jsonl")
    assert client.generate_json("s","u",SCHEMA)=={"ok":True}


def test_auth_error_is_not_retried(tmp_path):
    calls=0
    def handler(request):
        nonlocal calls;calls+=1;return httpx.Response(401,request=request)
    client=DeepSeekClient(api_key="x",transport=httpx.MockTransport(handler),audit_path=tmp_path/"audit.jsonl")
    with pytest.raises(DeepSeekError): client.generate_json("s","u",SCHEMA)
    assert calls==1

