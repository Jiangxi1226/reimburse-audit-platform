import os, base64
from openai import OpenAI


def _load_env():
    """应用进程不自动加载 .env，这里兜底：首次 import 时把项目根 .env 灌入 os.environ。

    否则 core/llm.py 的 os.getenv 拿不到 LLM_API_KEY/LLM_BASE_URL/LLM_TEXT_MODEL，
    构造 OpenAI() 时 api_key 为空，会抛 'Missing credentials'。
    用 setdefault 不覆盖已有环境变量（明确 export 的优先）。
    """
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'").strip()
                if k:
                    os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass


_load_env()

def _config() -> dict:
    """实时读取当前生效的 LLM 配置（每次新建客户端时取，而非首次 import 固定）。

    这样通过 /setup 或设置入口改写 .env 后，无需重启进程即可让后续请求用上新 key。
    直接读 os.environ 现值（reload_config 会用 os.environ[k]=v 覆盖），保证热更新生效。
    """
    return {
        "base_url": os.getenv("LLM_BASE_URL", ""),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", os.getenv("LLM_TEXT_MODEL", "")),
        "image_model": os.getenv("LLM_IMAGE_MODEL", ""),
    }


def reload_config():
    """强制重读 .env 并覆盖 os.environ 与模块级常量（供 /setup 与设置入口调用）。

    与 _load_env 的 setdefault 不同：这里用直接赋值，确保写入 .env 的新值立即生效，
    后续 LLM() 再取 _config() 即拿到新配置。
    """
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_TEXT_MODEL", "LLM_IMAGE_MODEL"):
        val = ""
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(key + "="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'").strip()
                        break
        except Exception:
            pass
        if val:
            os.environ[key] = val  # 覆盖（而非 setdefault），保证热更新
    return _config()


# ── 统一走 DeepSeek 一条链路（文本 + 多模态识图）────────────────────────────
# 环境变量命名沿用项目 .env 的 LLM_* 前缀；这里的 DEEPSEEK_* 仅为易读别名，值即 DeepSeek 端点。
# 本地 Qwen2-VL 与旧 Agnes 云端链路均已废弃删除，仅保留 DeepSeek 文本 + 识图。
DEEPSEEK_BASE_URL = os.getenv("LLM_BASE_URL", "")
DEEPSEEK_API_KEY = os.getenv("LLM_API_KEY", "")
DEEPSEEK_TEXT_MODEL = os.getenv("LLM_TEXT_MODEL", "")
DEEPSEEK_IMAGE_MODEL = os.getenv("LLM_IMAGE_MODEL", "")


class LLM:
    """统一 LLM 客户端：只负责 DeepSeek（OpenAI 兼容）的文本 / 识图调用。

    对外能力：
      chat(messages, ...)        纯文本对话
      chat_with_image(text, image)  多模态识图（图片走 DeepSeek 视觉模型）
      encode_image(path)         base64 编码图片
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        cfg = _config()
        self.model = model or cfg["model"]
        self.api_key = api_key or cfg["api_key"] or os.getenv("LLM_API_KEY", DEEPSEEK_API_KEY)
        self.base_url = base_url or cfg["base_url"] or os.getenv("LLM_BASE_URL", DEEPSEEK_BASE_URL)
        if not self.api_key:
            raise ValueError(
                "未配置 LLM_API_KEY。请在项目根 .env 中填写，或用设置入口提交；"
                "或先设置环境变量 LLM_API_KEY。/setup 页面填一次即可。")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)


    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature)
        return response.choices[0].message.content


    def chat_stream(self, messages: list[dict], temperature: float = 0.7):
        """流式对话：逐块 yield 文本增量(打字机效果)。供问答流式端点复用。

        走 OpenAI 兼容的 stream=True；每 chunk 的 delta.content 增量 yield。
        """
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


    def chat_with_image(self, text: str, image_input, temperature: float = 0.7) -> str:
        """识图：统一走 DeepSeek 多模态（DEEPSEEK_IMAGE_MODEL = LLM_IMAGE_MODEL）。

        image_input 支持三种形态：
          - 图片文件路径(str)
          - base64 编码字符串
          - 原始 bytes
        本地 Qwen2-VL 与旧 Agnes 云端链路已删除，只剩 DeepSeek 一条路径。
        """
        return self._chat_with_cloud_vision(text, image_input, temperature)

    def _chat_with_cloud_vision(self, text: str, image_input, temperature: float = 0.7) -> str:
        """DeepSeek 多模态识图：把图片拼进 OpenAI 兼容的 image_url 内容块。"""
        if isinstance(image_input, bytes):
            b64 = base64.b64encode(image_input).decode("utf-8")
        elif isinstance(image_input, str) and not self._is_base64_image(image_input):
            b64 = self.encode_image(image_input)
        else:
            b64 = image_input
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
        response = self.client.chat.completions.create(
            model=DEEPSEEK_IMAGE_MODEL, messages=[{"role": "user", "content": content}],
            temperature=temperature)
        return response.choices[0].message.content


    # ── 图片格式识别工具(供云端识图前校验/规范 base64 bytes)────────────────
    _IMAGE_SIGNATURES = [
        ('/9j',    '.jpg',  'JPEG'),
        ('iVBOR',  '.png',  'PNG'),
        ('R0lG',   '.gif',  'GIF'),
        ('UklGR',  '.webp', 'WebP'),
        ('Qk',     '.bmp',  'BMP'),
    ]

    _SUPPORTED_SUFFIXES = {suffix for _, suffix, _ in _IMAGE_SIGNATURES}

    @classmethod
    def _is_base64_image(cls, s: str) -> bool:
        """判断字符串是否是 base64 编码的图片（而非文件路径）。"""
        return any(s.startswith(prefix) for prefix, _, _ in cls._IMAGE_SIGNATURES)


    @staticmethod
    def encode_image(file_path: str) -> str:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
