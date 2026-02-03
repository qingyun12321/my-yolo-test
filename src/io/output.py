from __future__ import annotations

import json
import sys
from typing import TextIO


class JsonlEmitter:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def emit(self, record: dict) -> None:
        payload = json.dumps(record, ensure_ascii=True)
        self._stream.write(payload + "\n")
        self._stream.flush()

    def emit_many(self, records: list[dict]) -> None:
        for record in records:
            self.emit(record)
