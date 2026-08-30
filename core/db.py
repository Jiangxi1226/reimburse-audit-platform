import sqlite3, os, uuid, json, threading
from contextlib import contextmanager


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "knowledge_base.db")

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """获取当前线程的数据库连接（线程安全）。"""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(DB_DIR, exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def transaction():
    """事务上下文管理器——自动 commit/rollback。"""
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise



def init_db():
    """建表（幂等——IF NOT EXISTS）。"""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,          -- UUID
            username TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'user',      -- admin / user
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,          -- UUID
            user_id TEXT,                 -- 上传者
            filename TEXT NOT NULL,       -- 原始文件名
            file_type TEXT,               -- pdf/docx/xlsx/pptx/image/text
            file_size_bytes INTEGER,
            chunk_count INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active', -- active / deleted
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,          -- UUID
            user_id TEXT,                 -- 会话所属用户
            title TEXT,                   -- 会话标题（首条问题截取）
            message_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active', -- active / closed
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS error_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_code TEXT,              -- 错误码（NET_001 / DB_001 / LLM_001）
            error_type TEXT,              -- Python 异常类型
            message TEXT,                 -- 错误描述
            traceback TEXT,               -- 堆栈
            context TEXT,                 -- 上下文（用户输入、正在执行的操作）
            severity TEXT DEFAULT 'error',-- info / warning / error / critical
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,                -- 关联的 Agent 追踪 ID
            user_id TEXT,                 -- 发起用户
            tenant_id TEXT,               -- 租户（隔离）
            role TEXT,                    -- admin / analyst / user / guest
            protocol TEXT DEFAULT 'tool', -- tool / api / upload
            tool TEXT,                    -- 工具名
            action TEXT,                  -- 操作名
            args TEXT,                    -- 参数（已脱敏）
            risk TEXT,                    -- low / medium / high
            verdict TEXT,                 -- allow / confirm / deny
            ok INTEGER,                   -- 1=成功 0=失败
            result TEXT,                  -- 返回结果摘要（已脱敏）
            masked INTEGER DEFAULT 0,     -- 是否做了脱敏
            budget INTEGER DEFAULT 0,     -- 本次消耗预算
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_logs(tool, action);
        CREATE INDEX IF NOT EXISTS idx_audit_verdict ON audit_logs(verdict);

        CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_errors_code ON error_logs(error_code);
        CREATE INDEX IF NOT EXISTS idx_errors_created ON error_logs(created_at);
    """)



def create_user(username: str, role: str = "user") -> dict:
    with transaction() as conn:
        uid = str(uuid.uuid4())[:12]
        conn.execute("INSERT INTO users (id, username, role) VALUES (?, ?, ?)",
                     (uid, username, role))
        return {"id": uid, "username": username, "role": role}


def get_user(username: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None



def register_document(user_id: str, filename: str, file_type: str,
                      file_size: int, chunk_count: int = 0,
                      image_count: int = 0) -> str:
    with transaction() as conn:
        doc_id = str(uuid.uuid4())[:12]
        conn.execute(
            """INSERT INTO documents (id, user_id, filename, file_type, file_size_bytes, chunk_count, image_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, user_id, filename, file_type, file_size, chunk_count, image_count)
        )
        return doc_id


def get_document_stats() -> dict:
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM documents WHERE status='active'").fetchone()[0]
    by_type = conn.execute(
        "SELECT file_type, COUNT(*) FROM documents WHERE status='active' GROUP BY file_type"
    ).fetchall()
    return {
        "total_documents": total,
        "by_type": {r[0]: r[1] for r in by_type},
    }



def create_session(user_id: str = None, title: str = "") -> str:
    with transaction() as conn:
        sid = str(uuid.uuid4())[:12]
        conn.execute(
            "INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)",
            (sid, user_id, title[:100])
        )
        return sid


def update_session(session_id: str, message_count: int = None,
                   status: str = None):
    conn = _get_conn()
    if message_count is not None:
        conn.execute("UPDATE sessions SET message_count = ? WHERE id = ?",
                     (message_count, session_id))
    if status:
        conn.execute("UPDATE sessions SET status = ?, ended_at = CURRENT_TIMESTAMP WHERE id = ?",
                     (status, session_id))
    conn.commit()


def get_active_sessions(user_id: str = None) -> list[dict]:
    conn = _get_conn()
    query = "SELECT * FROM sessions WHERE status='active'"
    params = []
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    rows = conn.execute(query + " ORDER BY started_at DESC LIMIT 50", params).fetchall()
    return [dict(r) for r in rows]



def log_error(error_code: str, error_type: str, message: str,
              traceback_text: str = "", context: str = "",
              severity: str = "error"):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO error_logs (error_code, error_type, message, traceback, context, severity)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (error_code, error_type, message[:500], traceback_text[:2000], context[:500], severity)
    )
    conn.commit()


def log_audit(user_id: str, tool: str, action: str, args: str = "",
              risk: str = "low", verdict: str = "allow", ok: bool = True,
              result: str = "", tenant_id: str = "", role: str = "user",
              trace_id: str = "", protocol: str = "tool", masked: bool = False,
              budget: int = 0):
    """权限审计日志 —— 完整记录"谁 / 何时 / 想调什么 / 被放行还是拒绝 / 结果如何"。

    与 error_logs 不同：audit_logs 记录的是"工具调用的授权与处置过程"，
    用于事后追溯与安全审查（图「审计日志——完整记录操作与结果」）。
    """
    conn = _get_conn()
    # args 是 dict（来自 Guard._audit 传入的 safe_args），不能切片；序列化成 JSON 再截断，
    # 避免 dict[:800] 抛 KeyError 导致整个审计被 _audit 的 except 静默吞掉、日志丢失。
    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
    conn.execute(
        """INSERT INTO audit_logs
           (trace_id, user_id, tenant_id, role, protocol, tool, action,
            args, risk, verdict, ok, result, masked, budget)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trace_id or "", user_id or "", tenant_id or "", role or "user", protocol,
         tool, action, args_str[:800], risk, verdict, 1 if ok else 0,
         (result or "")[:1000], 1 if masked else 0, budget)
    )
    conn.commit()


def get_recent_audit(limit: int = 50, tool: str = None, verdict: str = None) -> list[dict]:
    conn = _get_conn()
    query = "SELECT * FROM audit_logs"
    params = []
    if tool:
        query += " WHERE tool = ?"
        params.append(tool)
    if verdict:
        query += (" AND verdict = ?" if tool else " WHERE verdict = ?")
        params.append(verdict)
    rows = conn.execute(query + " ORDER BY created_at DESC LIMIT ?", params + [limit]).fetchall()
    return [dict(r) for r in rows]


def get_recent_errors(limit: int = 20, severity: str = None) -> list[dict]:
    conn = _get_conn()
    query = "SELECT * FROM error_logs"
    params = []
    if severity:
        query += " WHERE severity = ?"
        params.append(severity)
    rows = conn.execute(query + " ORDER BY created_at DESC LIMIT ?", params + [limit]).fetchall()
    return [dict(r) for r in rows]


ERROR_CODES = {
    "NET_001": "LLM API 网络连接失败",
    "NET_002": "LLM API 响应超时",
    "NET_003": "文件上传网络中断",
    "DB_001": "ChromaDB 写入失败",
    "DB_002": "SQLite 查询失败",
    "DB_003": "向量索引重建失败",
    "LLM_001": "FC 模式调用失败（已降级 ReAct）",
    "LLM_002": "答案生成失败",
    "LLM_003": "HyDE 假设文档生成失败",
    "AGENT_001": "Agent 达到最大步数限制（部分完成）",
    "AGENT_002": "Agent 振荡循环被检测",
    "AGENT_003": "Agent 提前终止被拦截",
    "FILE_001": "文档格式不支持",
    "FILE_002": "文档文件大小超限",
    "FILE_003": "文档解析失败",
    "SEC_001": "内容安全扫描触发告警",
}


init_db()
