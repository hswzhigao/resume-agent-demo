"""输出侧 PII 打码：手机号、邮箱。默认开启，可在配置中关闭。"""

import re

_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"([A-Za-z0-9._%+-]{2,})@([A-Za-z0-9.-]+)")


def mask_pii(text: str) -> str:
    text = _PHONE.sub(lambda m: f"{m.group()[:3]}****{m.group()[-2:]}", text)
    text = _EMAIL.sub(lambda m: f"{m.group(1)[:2]}******@{m.group(2)}", text)
    return text
