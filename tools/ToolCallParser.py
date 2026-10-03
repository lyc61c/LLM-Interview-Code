"""纯 Python 的 Responses function-call 流状态机，不调用 SDK，不执行工具。

生命周期：output_item.added → 零或多个 arguments.delta → arguments.done。
output_index 区分交错的调用；完成时将完整 JSON 解析为参数。
这是三类事件的教学实现，不是所有供应商协议的通用流解析器。
"""

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    index: int
    item_id: str
    call_id: str
    name: str
    arguments: dict


@dataclass
class _PendingCall:
    item_id: str
    call_id: str
    name: str
    fragments: list = field(default_factory=list)
    result: ToolCall | None = None


class ToolCallParser:
    """feed(event) 返回刚完成的 ToolCall 或 None；finish() 返回按索引排序的列表。

    前提：event 为 Responses 风格的 dict，字段完整，参数为 JSON object。
    仅保留调用注册、追加、完成的状态检查；其他非工具事件被忽略。
    """

    def __init__(self):
        self._calls = {}

    def feed(self, event):
        event_type = event["type"]
        if event_type == "response.output_item.added":
            item = event["item"]
            if item.get("type") != "function_call":
                return None
            index = event["output_index"]
            if index in self._calls:
                raise ValueError(f"重复的 output_index: {index}")
            self._calls[index] = _PendingCall(
                item["id"], item["call_id"], item["name"], [item.get("arguments", "")],
            )
            return None

        if event_type not in ("response.function_call_arguments.delta", "response.function_call_arguments.done"):
            return None
        index = event["output_index"]
        if index not in self._calls:
            raise ValueError(f"未知 output_index: {index}，应先收到 added")
        call = self._calls[index]
        if call.result is not None:
            raise ValueError(f"output_index {index} 已经 done")
        if event_type.endswith(".delta"):
            call.fragments.append(event["delta"])
            return None

        # done 通常给出完整参数；没有该字段时使用 delta 拼接结果。
        arguments_text = event.get("arguments", "".join(call.fragments))
        arguments = json.loads(arguments_text)
        call.result = ToolCall(index, call.item_id, call.call_id, call.name, arguments)
        return call.result

    def finish(self):
        unfinished = [index for index, call in self._calls.items() if call.result is None]
        if unfinished:
            raise ValueError(f"流结束时仍有未完成的 function call: {unfinished}")
        return [self._calls[index].result for index in sorted(self._calls)]


def parse_tool_calls(events):
    """一次性消费事件 iterable，检查流完整性并返回 ToolCall 列表。"""
    parser = ToolCallParser()
    for event in events:
        parser.feed(event)
    return parser.finish()


if __name__ == "__main__":
    events = [
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "weather", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city":"上海"}'},
        {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": '{"city":"上海"}'},
    ]
    print(parse_tool_calls(events))
