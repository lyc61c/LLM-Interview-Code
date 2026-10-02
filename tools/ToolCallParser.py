"""纯 Python 的 Responses function-call 流状态机，不调用 SDK，不执行工具。

生命周期：output_item.added → 零或多个 arguments.delta → arguments.done。
output_index 区分交错的调用；完成时拼接 JSON 并验证为参数对象。
这是三类事件的教学实现，不是所有供应商协议的通用流解析器。
"""

import json
import math
from collections.abc import Mapping
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

    重复索引、未知索引、重复 done、done 后 delta、item_id 不匹配、非法 JSON
    会抛 ValueError。其他非工具事件被忽略，便于直接消费 Responses 流。
    """

    def __init__(self):
        self._calls = {}

    @staticmethod
    def _index(event):
        index = event.get("output_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("function-call 事件需要非负整数 output_index")
        return index

    def feed(self, event):
        if not isinstance(event, Mapping):
            raise TypeError("event 必须是 Mapping，如 dict")
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise ValueError("event 需要字符串 type")
        if event_type == "response.output_item.added":
            item = event.get("item")
            if not isinstance(item, Mapping):
                raise ValueError("output_item.added 需要 item 对象")
            if item.get("type") != "function_call":
                return None
            index = self._index(event)
            if index in self._calls:
                raise ValueError(f"重复的 output_index: {index}")
            if any(not isinstance(item.get(key), str) or not item[key] for key in ("id", "call_id", "name")):
                raise ValueError("function_call 需要非空 id、call_id、name")
            initial = item.get("arguments", "")
            if not isinstance(initial, str):
                raise ValueError("arguments 必须是 JSON 字符串")
            self._calls[index] = _PendingCall(item["id"], item["call_id"], item["name"], [initial])
            return None

        if event_type not in ("response.function_call_arguments.delta", "response.function_call_arguments.done"):
            return None
        index = self._index(event)
        if index not in self._calls:
            raise ValueError(f"未知 output_index: {index}，应先收到 added")
        call = self._calls[index]
        if "item_id" in event and event["item_id"] != call.item_id:
            raise ValueError(f"output_index {index} 的 item_id 不匹配")
        if call.result is not None:
            raise ValueError(f"output_index {index} 已经 done")
        if event_type.endswith(".delta"):
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise ValueError("delta 必须是字符串")
            call.fragments.append(delta)
            return None

        accumulated = "".join(call.fragments)
        arguments_text = event.get("arguments", accumulated)
        if not isinstance(arguments_text, str):
            raise ValueError("done.arguments 必须是 JSON 字符串")
        if accumulated and arguments_text != accumulated:
            raise ValueError(f"output_index {index} 的 done.arguments 与 delta 拼接结果不一致")
        try:
            # 禁止非标准 NaN/Infinity，并拒绝合法 JSON 数字转为 Python float 后溢出。
            arguments = json.loads(arguments_text, parse_constant=self._reject_constant,
                                   parse_float=self._finite_float)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"output_index {index} 的参数 JSON 无效: {error}") from error
        if not isinstance(arguments, dict):
            raise ValueError(f"output_index {index} 的参数必须是 JSON object")
        call.result = ToolCall(index, call.item_id, call.call_id, call.name, arguments)
        return call.result

    @staticmethod
    def _reject_constant(value):
        raise ValueError(f"非标准 JSON 数值: {value}")

    @staticmethod
    def _finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"JSON 数值超出有限 Python float 支持范围: {value}")
        return result

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
