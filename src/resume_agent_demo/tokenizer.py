"""轻量分词：ASCII 词（小写）+ CJK 单字与相邻二元组。

不做中文分词模型，目标是确定性、无依赖；用于本地 hash 向量与关键词覆盖率打分。
"""

import re

_ASCII = re.compile(r"[A-Za-z0-9_+#.]+")


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _ASCII.finditer(text):
        tokens.append(m.group().lower())

    run: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            run.append(ch)
        else:
            if run:
                tokens.extend(_cjk_tokens(run))
                run = []
    if run:
        tokens.extend(_cjk_tokens(run))
    return tokens


def _cjk_tokens(chars: list[str]) -> list[str]:
    out: list[str] = list(chars)
    out.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return out
