from resume_agent_demo.pii import mask_pii


def test_masks_cn_mobile():
    out = mask_pii("电话 13811643209 请联系")
    assert "13811643209" not in out
    assert "138****09" in out


def test_masks_email():
    out = mask_pii("邮箱 ccuzhigao@163.com 发简历")
    assert "ccuzhigao@163.com" not in out
    assert "cc******@163.com" in out


def test_plain_text_unchanged():
    assert mask_pii("无敏感信息，Java 高并发") == "无敏感信息，Java 高并发"
