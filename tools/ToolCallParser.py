"""按 output_index 收集工具调用的 JSON 片段，完成后解析为参数。"""

import json


class ToolCallParser:
    """用普通字典保存调用信息；event 使用 Responses 风格的完整字段。"""

    def __init__(self):
        self.calls = {}
        self.results = {}

    def feed(self, event):
        event_type = event["type"]
        if event_type == "response.output_item.added":
            item = event["item"]
            if item.get("type") != "function_call":
                return None
            index = event["output_index"]
            self.calls[index] = {
                "index": index,
                "item_id": item["id"],
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": item.get("arguments", ""),
            }
            return None

        if event_type not in ("response.function_call_arguments.delta", "response.function_call_arguments.done"):
            return None
        index = event["output_index"]
        call = self.calls[index]
        if event_type.endswith(".delta"):
            call["arguments"] += event["delta"]
            return None

        # done 通常给出完整参数；没有该字段时使用 delta 拼接结果。
        arguments_text = event.get("arguments", call["arguments"])
        call["arguments"] = json.loads(arguments_text)
        self.results[index] = call
        return call

    def finish(self):
        """返回按索引排序的已完成调用列表。"""
        return [self.results[index] for index in sorted(self.results)]


def parse_tool_calls(events):
    """消费事件列表，返回解析完成的调用字典列表；不执行工具。"""
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
