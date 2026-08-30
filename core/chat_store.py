"""会话历史持久化存储——用 JSON 文件跨请求保存聊天记录。

Gradio 的 chatbot 历史只存在于单次页面会话，刷新即丢。
这里将每条问答追加到 JSON 文件，供"最近会话"列表读取、切换恢复。
"""
import json, os, time, uuid, threading

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat")
STORE_PATH = os.path.join(DATA_DIR, "sessions.json")

_lock = threading.Lock()


def _load() -> dict:
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _display_title(s: dict) -> str:
    msgs = s.get("messages", [])
    title = s.get("title") or ""
    if not title:
        for m in msgs:
            if m["role"] == "user":
                title = m["content"][:20]
                break
    return title or "新会话"


def list_sessions(limit: int = 30) -> list[dict]:
    """按最近更新排序返回会话列表。"""
    with _lock:
        data = _load()
        sessions = []
        for sid, s in data.items():
            msgs = s.get("messages", [])
            user_count = sum(1 for m in msgs if m["role"] == "user")
            sessions.append({
                "id": sid,
                "title": _display_title(s),
                "preview": (msgs[-1]["content"][:20] if msgs else "暂无消息"),
                "count": user_count,
                "updated": s.get("updated", 0),
            })
        sessions.sort(key=lambda x: -x["updated"])
        return sessions[:limit]


def create_session() -> str:
    """新建一个会话，返回 session_id。"""
    with _lock:
        data = _load()
        sid = str(uuid.uuid4())[:12]
        data[sid] = {
            "title": "", "messages": [],
            "created": time.time(), "updated": time.time(),
        }
        _save(data)
        return sid


def get_messages(sid: str) -> list[dict]:
    """返回会话历史消息列表 [{role, content}, ...]。"""
    with _lock:
        data = _load()
        s = data.get(sid)
        if not s:
            return []
        return [{"role": m["role"], "content": m["content"]} for m in s.get("messages", [])]


def save_exchange(sid: str, user_msg: str, bot_msg: str):
    """保存一轮问答；首条问题时自动生成会话标题。"""
    with _lock:
        data = _load()
        s = data.get(sid)
        if not s:
            return
        s.setdefault("messages", [])
        s["messages"].append({"role": "user", "content": user_msg})
        if bot_msg:
            s["messages"].append({"role": "assistant", "content": bot_msg})
        s["updated"] = time.time()
        if not s.get("title"):
            s["title"] = user_msg[:20]
        _save(data)


def delete_session(sid: str):
    """删除一个会话。"""
    with _lock:
        data = _load()
        if sid in data:
            del data[sid]
            _save(data)
