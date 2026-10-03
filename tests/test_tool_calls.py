from tools.ToolCallParser import ToolCallParser, parse_tool_calls


def added(index, name):
    return {"type": "response.output_item.added", "output_index": index,
            "item": {"type": "function_call", "id": f"fc_{index}",
                     "call_id": f"call_{index}", "name": name, "arguments": ""}}


def delta(index, text):
    return {"type": "response.function_call_arguments.delta", "output_index": index, "delta": text}


def done(index):
    return {"type": "response.function_call_arguments.done", "output_index": index}


def test_interleaved_json_fragments():
    events = [added(1, "search"), added(0, "weather"),
              delta(1, '{"q":"'), delta(0, '{"city":"上海"}'),
              delta(1, 'LLM"}'), done(1), done(0)]
    results = parse_tool_calls(events)
    assert [call["index"] for call in results] == [0, 1]
    assert results[0]["arguments"] == {"city": "上海"}
    assert results[1]["name"] == "search" and results[1]["arguments"] == {"q": "LLM"}


def test_done_can_provide_complete_json():
    parser = ToolCallParser()
    parser.feed(added(0, "example"))
    result = parser.feed({"type": "response.function_call_arguments.done", "output_index": 0,
                          "arguments": '{"number":3}'})
    assert result["arguments"] == {"number": 3}
    assert parser.finish() == [result]
