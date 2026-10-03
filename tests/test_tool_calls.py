import pytest

from tools.ToolCallParser import ToolCallParser, parse_tool_calls


def added(index, name="weather"):
    return {"type": "response.output_item.added", "output_index": index,
            "item": {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
                     "name": name, "arguments": ""}}


def delta(index, text):
    return {"type": "response.function_call_arguments.delta", "output_index": index,
            "item_id": f"fc_{index}", "delta": text}


def done(index, text):
    return {"type": "response.function_call_arguments.done", "output_index": index,
            "item_id": f"fc_{index}", "arguments": text}


def test_interleaved_tool_calls_are_complete_and_ordered_by_index():
    events = [added(2, "search"), added(0),
              delta(2, '{"q":"'), delta(0, '{"city":'),
              delta(2, 'LLM"}'), done(2, '{"q":"LLM"}'),
              {"type": "response.output_text.delta", "delta": "Thinking"},
              delta(0, '"上海"}'), done(0, '{"city":"上海"}')]
    results = parse_tool_calls(events)
    assert [call.index for call in results] == [0, 2]
    assert results[0].name == "weather" and results[0].arguments == {"city": "上海"}
    assert results[1].call_id == "call_2" and results[1].arguments == {"q": "LLM"}


def test_done_can_supply_all_arguments_if_no_deltas_were_received():
    parser = ToolCallParser()
    assert parser.feed(added(0)) is None
    result = parser.feed(done(0, '{"x":1}'))
    assert result.arguments == {"x": 1}
    assert parser.finish() == [result]
    assert parse_tool_calls([]) == []


@pytest.mark.parametrize("text", ['{"x":', ""])
def test_tool_calls_reject_unparseable_json(text):
    with pytest.raises(ValueError):
        parse_tool_calls([added(0), done(0, text)])


def test_tool_calls_accept_large_finite_float_arguments():
    result = parse_tool_calls([added(0), done(0, '{"number":1e308}')])
    assert result[0].arguments == {"number": 1e308}


def test_tool_calls_reject_unfinished_stream():
    with pytest.raises(ValueError, match="未完成"):
        parse_tool_calls([added(0), delta(0, '{"x":')])


def test_tool_calls_use_final_done_arguments_or_delta_fallback():
    result = parse_tool_calls([added(0), delta(0, '{"x":'), done(0, '{"x":2}')])
    assert result[0].arguments == {"x": 2}
    event = {"type": "response.function_call_arguments.done", "output_index": 0}
    result = parse_tool_calls([added(0), delta(0, '{"x":1}'), event])
    assert result[0].arguments == {"x": 1}


@pytest.mark.parametrize("events", [
    [delta(0, "{}")], [done(0, "{}")], [added(0), added(0)],
    [added(0), done(0, "{}"), done(0, "{}")],
    [added(0), done(0, "{}"), delta(0, " ")],
])
def test_tool_calls_reject_unknown_duplicate_and_completed_indices(events):
    with pytest.raises(ValueError):
        parse_tool_calls(events)


def test_tool_calls_ignore_nonfunction_items():
    parser = ToolCallParser()
    assert parser.feed({"type": "response.output_item.added", "item": {"type": "message"}}) is None
