from __future__ import annotations

from app.schemas import TicketDraft


def make_ticket_draft(question: str) -> TicketDraft:
    """A deterministic editable draft. No database write or model call occurs here."""
    text = question.strip()
    if any(word in text for word in ("密码", "账号", "登录")):
        category = "账号"
    elif any(word in text.lower() for word in ("vpn", "网络", "连接")):
        category = "网络"
    elif any(word in text for word in ("电脑", "设备", "维修")):
        category = "设备"
    elif any(word in text for word in ("报销", "发票", "费用")):
        category = "财务"
    else:
        category = "其他"
    return TicketDraft(title=text[:100], description=text, category=category)
