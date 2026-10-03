import pytest

from grep_first_agent.contract import ContractError, parse_result

JSON = '{"intent": "find", "pointers": {"task": [{"id": "t1", "name": "X", "confidence": 0.9}]}, "confidence": 0.8}'


def test_parse_from_content_blocks():
    blocks = [{"type": "thinking", "thinking": "{ignore}"}, {"type": "text", "text": JSON}]
    assert parse_result(blocks).pointers["task"][0].id == "t1"


def test_parse_json_inside_prose():
    r = parse_result(f"Sure! {{not json}} here you go:\n```json\n{JSON}\n```\nHope that helps")
    assert r.intent == "find" and r.confidence == 0.8


def test_contract_error():
    for bad in ["no json at all", '{"intent": 5, "pointers": "x"}', [{"type": "text", "text": "{"}], ""]:
        with pytest.raises(ContractError):
            parse_result(bad)
