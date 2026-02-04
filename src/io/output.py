from __future__ import annotations

"""输出模块：JSONL 记录器。"""

import json
import sys
from typing import TextIO


class JsonlEmitter:
    """JSONL 输出器（逐行 JSON）。"""

    def __init__(self, stream: TextIO | None = None) -> None:
        """初始化输出流。

        参数:
            stream: 目标输出流，默认为 stdout。
        """
        self._stream = stream or sys.stdout

    def emit(self, record: dict) -> None:
        """输出单条记录。

        参数:
            record: 需要输出的字典对象。
        """
        payload = json.dumps(record, ensure_ascii=True)
        self._stream.write(payload + "\n")
        self._stream.flush()

    def emit_many(self, records: list[dict]) -> None:
        """输出多条记录。

        参数:
            records: 记录列表。
        """
        for record in records:
            self.emit(record)
