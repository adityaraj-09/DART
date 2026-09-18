"""Push vs credit-gated decode on one producer.

    python -m dart experiment --seconds 3 --max-tokens 256
"""

from __future__ import annotations

import asyncio
import json

from dart.experiment import compare


def main() -> None:
    report = asyncio.run(compare(duration_s=3.0, max_tokens=256))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
