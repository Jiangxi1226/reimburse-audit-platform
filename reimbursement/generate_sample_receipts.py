# -*- coding: utf-8 -*-
"""生成样例发票图（用于端到端测试 OCR→结构化→审核）。

借鉴 expense-ai generate_receipts.py 的思路：程序化生成票据图片，
这样测试不依赖真实发票文件，且可控（每种票据可设计"应被判合规/违规"）。
用 PIL 画中文文本发票。
"""
import os
from datetime import date, timedelta
from PIL import Image, ImageDraw, ImageFont


def _d(days_ago: int) -> str:
    """生成"今天-N天"的日期。

    票据日期若硬编码会随时间过期：图片是 2026-08-30 生成的，日期写死 2026-08-01，
    一个月后再跑端到端就全部超出 30 天报销窗口 → 本应 approve 的被判 manual_review。
    """
    return (date.today() - timedelta(days=days_ago)).isoformat()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_THIS_DIR, "sample_receipts")
os.makedirs(_OUT, exist_ok=True)


def _font(size: int):
    # Windows 常见中文字体；无则回退默认（可能乱码，仅示意）
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\msyhbd.ttc"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_receipt(filename: str, lines: list[str], width=520):
    img = Image.new("RGB", (width, 300 + len(lines) * 34), "white")
    d = ImageDraw.Draw(img)
    f = _font(20)
    d.text((20, 20), lines[0], fill="black", font=_font(24))
    y = 80
    for ln in lines[1:]:
        d.text((30, y), ln, fill="black", font=f)
        y += 34
    return img


def main():
    samples = {
        "住宿_北京_超标.png": [
            "XX酒店住宿发票  №05001234",
            f"客户：张伟    日期：{_d(2)}",
            "地址：北京市朝阳区建国路88号",
            "房间：标准间 1 晚",
            "住宿费用：¥620.00",
            "合计（价税）：¥620.00",
        ],
        "餐饮_商务宴请.png": [
            "XX餐厅餐饮发票  №06005678",
            f"日期：{_d(2)}",
            "晚餐 1 桌",
            "金额：¥250.00",
        ],
        "交通_地铁.png": [
            "XX交通电子发票  №07001111",
            f"日期：{_d(3)}",
            "业务：地铁通勤",
            "金额：¥50.00",
        ],
        "个人消费_护肤.png": [
            "XX商场购物小票  №08002222",
            f"日期：{_d(3)}",
            "面膜 / 护肤品",
            "金额：¥300.00",
        ],
    }
    for name, lines in samples.items():
        img = _draw_receipt(name, lines)
        path = os.path.join(_OUT, name)
        img.save(path)
        print("OK", path)


if __name__ == "__main__":
    main()
