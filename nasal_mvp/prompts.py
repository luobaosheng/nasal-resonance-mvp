from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output"
BANK = OUT / "prompt_bank.json"

from .corpus import texts as corpus_texts

DEFAULTS = corpus_texts() + [
    "啊——（持续约 3 秒）",
    "衣——（持续约 3 秒）",
    "乌——（持续约 3 秒）",
]


def load_bank() -> list[str]:
    if BANK.exists():
        try:
            data = json.loads(BANK.read_text(encoding="utf-8"))
            extra = [str(x).strip() for x in data if str(x).strip()]
            seen = list(DEFAULTS)
            for x in extra:
                if x not in seen:
                    seen.append(x)
            return seen
        except json.JSONDecodeError:
            pass
    return list(DEFAULTS)


def add_prompt(text: str) -> list[str]:
    t = (text or "").strip()
    bank = load_bank()
    if t and t not in bank:
        bank.append(t)
        OUT.mkdir(parents=True, exist_ok=True)
        custom = [x for x in bank if x not in DEFAULTS]
        BANK.write_text(json.dumps(DEFAULTS + custom, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_bank()
