from __future__ import annotations

from search_agent.infrastructure.receipts import write_receipt


class JsonReceiptWriter:
    def write(self, report, output_dir: str) -> str:
        return write_receipt(report, output_dir=output_dir)
