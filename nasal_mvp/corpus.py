from __future__ import annotations

# 中文儿童 VPI 客观声学评估语句库 v1.0（草案，须临床合作者审定）
# 单段录音内同时含口腔目标与鼻音基线时，才能做自参照。

SENTENCES = [
    {
        "id": "VPI_01",
        "text": "爸爸打靶",
        "category": "口腔爆破音测试",
        "target_chars": "爸打靶",
        "baseline_chars": "",
        "description": "高压口腔爆破音，测气流鼻腔泄露",
    },
    {
        "id": "VPI_02",
        "text": "哥哥骑车",
        "category": "口腔爆破音测试",
        "target_chars": "哥骑车",
        "baseline_chars": "",
        "description": "中后位塞音及擦音",
    },
    {
        "id": "VPI_03",
        "text": "大妈买大马",
        "category": "口鼻混合自参照",
        "target_chars": "大",
        "baseline_chars": "妈买马",
        "description": "核心锚点：大(口) vs 妈买马(鼻)",
    },
    {
        "id": "VPI_04",
        "text": "爸爸吃面",
        "category": "口鼻混合自参照",
        "target_chars": "爸吃",
        "baseline_chars": "面",
        "description": "核心锚点：爸/吃 vs 面",
    },
    {
        "id": "VPI_05",
        "text": "阿姨吃鱼",
        "category": "持续高元音测试",
        "target_chars": "阿姨吃鱼",
        "baseline_chars": "",
        "description": "高元音 F1/F2 与高频衰减",
    },
    {
        "id": "VPI_06",
        "text": "乌鸦喝水",
        "category": "持续高元音测试",
        "target_chars": "乌鸦喝水",
        "baseline_chars": "",
        "description": "圆唇高元音",
    },
    {
        "id": "VPI_07",
        "text": "大皮球",
        "category": "短句联动测试",
        "target_chars": "大皮球",
        "baseline_chars": "",
        "description": "双唇爆破音",
    },
    {
        "id": "VPI_08",
        "text": "开花朵",
        "category": "短句联动测试",
        "target_chars": "开花朵",
        "baseline_chars": "",
        "description": "清擦音与舌尖爆破音",
    },
    {
        "id": "VPI_09",
        "text": "妈妈买花",
        "category": "全鼻音基准对照",
        "target_chars": "花",
        "baseline_chars": "妈买",
        "description": "鼻音基线 + 花(口)",
    },
    {
        "id": "VPI_10",
        "text": "奶奶看书",
        "category": "全鼻音基准对照",
        "target_chars": "看书",
        "baseline_chars": "奶",
        "description": "鼻辅音 /n/ 对照",
    },
    {
        "id": "VPI_11",
        "text": "爸爸怕打靶，妈妈买牛奶",
        "category": "口鼻混合自参照",
        "target_chars": "爸怕打靶",
        "baseline_chars": "妈奶",
        "description": "演示句：口腔爆破 vs 妈奶",
    },
]

TARGET_CHARS = frozenset("爸怕打靶哥骑车大吃阿姨鱼乌鸦喝水皮球开花朵看书")
BASELINE_CHARS = frozenset("妈买马面奶奶")


def texts() -> list[str]:
    return [s["text"] for s in SENTENCES]


def match_sentence(prompt: str) -> dict | None:
    t = (prompt or "").replace("，", "").replace(" ", "").strip()
    for s in SENTENCES:
        key = s["text"].replace("，", "").replace(" ", "")
        if key and (key in t or t in key or t == key):
            return s
    return None


def expected_from_prompt(prompt: str) -> tuple[str, str]:
    s = match_sentence(prompt)
    if s:
        return s.get("target_chars") or "", s.get("baseline_chars") or ""
    return "".join(c for c in (prompt or "") if c in TARGET_CHARS), "".join(
        c for c in (prompt or "") if c in BASELINE_CHARS
    )
