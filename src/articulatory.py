"""Articulatory feedback generator.

Provides concrete tongue / lip / jaw adjustments for common Mandarin-L1
English pronunciation errors. Falls back to a panphon-driven generic tip
for phoneme pairs that are not in the curated list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from panphon import FeatureTable

    _ft = FeatureTable()
except Exception:  # pragma: no cover - panphon data may be missing in minimal envs
    _ft = None


@dataclass(frozen=True)
class ArticulatoryTip:
    description: str
    tongue: str
    lips: str
    jaw: str
    practice: str


TIPS: dict[str, ArticulatoryTip] = {
    "voiceless th → s": ArticulatoryTip(
        description="你把 /θ/ 发成了 /s/，舌尖缩回去了。",
        tongue="舌尖伸到上下齿之间，轻轻咬住。",
        lips="嘴唇自然向两侧拉开，不要收圆。",
        jaw="下巴微微放松，不要前伸。",
        practice="think, three, bath, mouth",
    ),
    "voiceless th → t": ArticulatoryTip(
        description="你把 /θ/ 发成了 /t/，气流被挡住了。",
        tongue="舌尖放到牙齿边缘，让气流从齿缝摩擦出来。",
        lips="嘴唇微张，保持放松。",
        jaw="下巴自然张开，比 /t/ 时略宽。",
        practice="think, thumb, birthday, nothing",
    ),
    "voiced th → z": ArticulatoryTip(
        description="你把 /ð/ 发成了 /z/，舌尖位置太靠后。",
        tongue="舌尖伸到上下齿之间，声带振动。",
        lips="嘴唇微微展开。",
        jaw="下巴保持中立，放松。",
        practice="this, that, mother, brother",
    ),
    "voiced th → d": ArticulatoryTip(
        description="你把 /ð/ 发成了 /d/，舌尖弹开了。",
        tongue="舌尖轻触牙齿边缘，不要让舌尖离开齿龈。",
        lips="嘴唇自然，不要用力。",
        jaw="下巴微微张开，保持稳定。",
        practice="this, then, weather, smooth",
    ),
    "V → W": ArticulatoryTip(
        description="你把 /v/ 发成了 /w/，没有用上齿咬唇。",
        tongue="舌头放平，不要抬起。",
        lips="上牙轻咬下唇，然后让气流从唇间通过。",
        jaw="下巴放松。",
        practice="very, video, have, love",
    ),
    "W → V": ArticulatoryTip(
        description="你把 /w/ 发成了 /v/，嘴唇先圆拢再快速打开。",
        tongue="舌头后部抬起，但不要碰到上颚。",
        lips="双唇圆拢成一个小圆，然后滑向后面的元音。",
        jaw="下巴稍向前突出。",
        practice="water, we, want, always",
    ),
    "R → L": ArticulatoryTip(
        description="你把 /r/ 发成了 /l/，舌尖顶到齿龈了。",
        tongue="舌尖悬空，舌身两侧贴住上臼齿，舌面中部下凹。",
        lips="嘴唇微微收圆。",
        jaw="下巴保持中立。",
        practice="red, right, sorry, very",
    ),
    "L → R": ArticulatoryTip(
        description="你把 /l/ 发成了 /r/，舌尖没有顶上去。",
        tongue="舌尖顶住上齿龈，气流从舌头两侧出来。",
        lips="嘴唇自然展开。",
        jaw="下巴微微张开。",
        practice="light, love, hello, school",
    ),
    "SH → S": ArticulatoryTip(
        description="你把 /ʃ/ 发成了 /s/，舌位太平。",
        tongue="舌身抬高，舌尖向后卷向硬腭前部。",
        lips="嘴唇微微收圆、向前突出。",
        jaw="下巴微微收紧。",
        practice="ship, she, show, washing",
    ),
    "CH → TS": ArticulatoryTip(
        description="你把 /tʃ/ 发成了 /ts/，舌位太靠前。",
        tongue="舌身前部抵住硬腭，然后释放。",
        lips="嘴唇微微收圆。",
        jaw="下巴保持稳定。",
        practice="chair, child, teacher, church",
    ),
    "J → DZ": ArticulatoryTip(
        description="你把 /dʒ/ 发成了 /dz/，舌位太靠前。",
        tongue="舌身前部抵住硬腭，声带振动，释放时带摩擦。",
        lips="嘴唇微微收圆。",
        jaw="下巴放松。",
        practice="juice, job, age, bridge",
    ),
    "Z → S devoicing": ArticulatoryTip(
        description="你把 /z/ 发成了 /s/，声带没有振动。",
        tongue="舌尖放在齿龈上，位置不变。",
        lips="嘴唇自然。",
        jaw="下巴放松，同时让声带振动。",
        practice="zoo, zero, easy, dogs",
    ),
    "S → Z voicing": ArticulatoryTip(
        description="你把 /s/ 发成了 /z/，声带振动过头了。",
        tongue="舌尖放在齿龈上，位置不变。",
        lips="嘴唇自然。",
        jaw="下巴放松，气流摩擦要大，声带不振动。",
        practice="sun, see, bus, house",
    ),
    "D → T devoicing": ArticulatoryTip(
        description="你把 /d/ 发成了 /t/，结尾没有振动。",
        tongue="舌尖抵住齿龈。",
        lips="嘴唇自然。",
        jaw="下巴放松，保持声带振动到结尾。",
        practice="bad, bed, road, decide",
    ),
    "B → P devoicing": ArticulatoryTip(
        description="你把 /b/ 发成了 /p/，结尾没有振动。",
        tongue="舌头放平。",
        lips="双唇闭合后快速打开，声带振动。",
        jaw="下巴放松。",
        practice="big, job, rabbit, about",
    ),
    "G → K devoicing": ArticulatoryTip(
        description="你把 /g/ 发成了 /k/，结尾没有振动。",
        tongue="舌根抬起抵住软腭。",
        lips="嘴唇自然。",
        jaw="下巴放松，声带振动。",
        practice="go, big, dog, again",
    ),
    "A → E (mouth too closed)": ArticulatoryTip(
        description="你把 /æ/ 发成了 /ɛ/，嘴巴张得不够大。",
        tongue="舌头平放，舌尖轻触下齿。",
        lips="嘴唇向两侧拉开。",
        jaw="下巴明显下沉，嘴巴张大。",
        practice="cat, bad, happy, family",
    ),
    "E → A (mouth too open)": ArticulatoryTip(
        description="你把 /ɛ/ 发成了 /æ/，嘴巴张得太大。",
        tongue="舌位比 /æ/ 稍高。",
        lips="嘴唇半开，横向拉开。",
        jaw="下巴下沉幅度比 /æ/ 小。",
        practice="bed, head, said, many",
    ),
    "I → EE (tense)": ArticulatoryTip(
        description="你把 /ɪ/ 发成了 /i/，肌肉太紧张。",
        tongue="舌位比 /i/ 低且靠后，放松。",
        lips="嘴唇微微张开，不要咧嘴。",
        jaw="下巴放松。",
        practice="ship, sit, milk, busy",
    ),
    "EE → I (lax)": ArticulatoryTip(
        description="你把 /i/ 发成了 /ɪ/，舌位不够高。",
        tongue="舌尖抵下齿，舌前部抬高。",
        lips="嘴角向两侧咧开。",
        jaw="下巴几乎不动。",
        practice="see, eat, tree, happy",
    ),
    "U → OO (rounded)": ArticulatoryTip(
        description="你把 /ʊ/ 发成了 /u/，嘴唇收得太紧。",
        tongue="舌位比 /u/ 低。",
        lips="双唇圆拢但放松，不要噘得太紧。",
        jaw="下巴微微下沉。",
        practice="good, book, look, could",
    ),
    "OO → U (unrounded)": ArticulatoryTip(
        description="你把 /u/ 发成了 /ʊ/，嘴唇不够紧。",
        tongue="舌后部抬高。",
        lips="双唇收圆、向前突出。",
        jaw="下巴微微收紧。",
        practice="food, moon, blue, do",
    ),
    "extra sound": ArticulatoryTip(
        description="你多加了一个音。",
        tongue="检查目标单词的音节结构。",
        lips="保持口型稳定，不要滑动。",
        jaw="下巴放松，减少多余动作。",
        practice="slowly, carefully",
    ),
    "missing sound": ArticulatoryTip(
        description="你漏掉了一个音。",
        tongue="把每个音都发到位再过渡到下一个。",
        lips="保持口型到发音结束。",
        jaw="下巴动作要完整。",
        practice="world, text, crisp",
    ),
}


_FEATURE_HINTS = {
    "constricted glottis": "声门收紧",
    "spread glottis": "声门打开，送气",
    "voice": "声带振动",
    "labial": "嘴唇参与",
    "round": "圆唇",
    "coronal": "舌尖/舌前部抬起",
    "anterior": "舌头靠前",
    "distributed": "舌面展开",
    "dorsal": "舌后部抬起",
    "high": "舌位抬高",
    "low": "舌位降低",
    "front": "舌头靠前",
    "back": "舌头靠后",
    "tense": "肌肉紧张",
    "long": "拉长音",
    "nasal": "气流从鼻子出",
}


def _generic_tip(expected: str, actual: str) -> ArticulatoryTip:
    """Build a generic tip by comparing panphon feature vectors."""
    if _ft is None:
        return ArticulatoryTip(
            description=f"目标音 {expected} 和你说出的 {actual} 不一致。",
            tongue="尝试模仿目标音的舌位。",
            lips="注意嘴唇形状。",
            jaw="注意下巴开合。",
            practice="listen and repeat",
        )

    exp_vec = _ft.fts(expected)
    act_vec = _ft.fts(actual)
    if not exp_vec or not act_vec:
        return ArticulatoryTip(
            description=f"目标音 {expected} 和你说出的 {actual} 不一致。",
            tongue="尝试模仿目标音的舌位。",
            lips="注意嘴唇形状。",
            jaw="注意下巴开合。",
            practice="listen and repeat",
        )

    diff = []
    for name, hint in _FEATURE_HINTS.items():
        ev = exp_vec[0].get(name)
        av = act_vec[0].get(name)
        if ev is not None and av is not None and ev != av:
            if ev == "+":
                diff.append(f"需要{hint}")
            elif ev == "-":
                diff.append(f"避免{hint}")

    detail = "；".join(diff[:4]) or "注意舌位、唇形和下巴的配合"
    return ArticulatoryTip(
        description=f"你说出的 {actual} 和目标 {expected} 在发音特征上不同。",
        tongue=detail,
        lips="观察 native speaker 的口型并模仿。",
        jaw="配合元音开合调整下巴。",
        practice=f"{expected} vs {actual}",
    )


def get_tip(expected: str | None, actual: str | None, label: str) -> dict:
    """Return a concrete articulatory tip as a plain dict.

    Parameters
    ----------
    expected : str | None
        Target IPA phone (None for insertions).
    actual : str | None
        Learner IPA phone (None for deletions).
    label : str
        Pre-computed error label from the analyzer.
    """
    tip: ArticulatoryTip
    if label in TIPS:
        tip = TIPS[label]
    elif expected and actual:
        tip = _generic_tip(expected, actual)
    elif label.startswith("extra") or actual:
        tip = TIPS["extra sound"]
    else:
        tip = TIPS["missing sound"]

    return {
        "description": tip.description,
        "tongue": tip.tongue,
        "lips": tip.lips,
        "jaw": tip.jaw,
        "practice": tip.practice,
    }


def attach_tips(analysis: dict) -> dict:
    """Mutate an analysis dict in place, adding tips to every error."""
    for word in analysis.get("words", []):
        for err in word.get("errors", []):
            err["tips"] = get_tip(
                err.get("expected"),
                err.get("actual"),
                err.get("label", ""),
            )
    return analysis
