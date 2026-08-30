"""确定性向量存储后端——sqlite(持久层,即时落盘) + faiss(内存加速)。

为什么不用 ChromaDB:
- chromadb 1.x 用 Rust 后台 System 做**异步 compaction**,HNSW 段文件只在 compaction 完成后
  才写盘,且**没有公开 flush/wait API**。入库写入返回 success 后若进程被强杀/立即重启/崩溃,
  段文件丢失、只剩 sqlite 元数据,重启检索报 "Error loading hnsw index"——反复损坏的温床。
- 本机无 MSVC C++ 编译器,**chromadb 0.x(依赖 chroma-hnswlib 源码编译)无法安装**,无法靠
  降级规避。
- 因此改用**确定性落盘**:所有向量/文本/元数据实时写入 sqlite(WAL,事务提交即落盘),
  faiss 索引仅作内存近似检索、每次改动后从 sqlite 重建——任何时刻强杀/重启,数据都在 sqlite,
  重启从 sqlite 全量重建索引,零丢失、零损坏。几千 chunk 量级下重建成本数十毫秒,完全可负担。

对外契约与旧 ChromaStore 保持一致(供 RAGPipeline / IndexSyncManager 等调用):
- add(id, vector, text) / add_batch(ids, vectors, texts, metadatas)
- search(query_vector, top_k=5, where=None) -> [{"id","text","similarity"}]
- count() / _data(兼容) / .collection.get()/.collection.delete(ids=/where=)
"""

import os, json, sqlite3, threading
import numpy as np
import faiss


# ---------------------------------------------------------------------------
# 模块级:单个 sqlite 连接 + 全局写锁(所有 collection 共用同一 DB 文件)
# ---------------------------------------------------------------------------
_DB_PATH = None
_DB = None
_DB_LOCK = threading.RLock()


def _get_db(persist_dir: str) -> sqlite3.Connection:
    """按目录建一个全局单连接(同目录多 ChromaStore 复用),开 WAL,建表。"""
    global _DB, _DB_PATH
    db_path = os.path.join(persist_dir, "vector_store.db")
    if _DB is not None and _DB_PATH == db_path:
        return _DB
    os.makedirs(persist_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS vectors (
            collection TEXT NOT NULL,
            id TEXT NOT NULL,
            vector BLOB NOT NULL,
            text TEXT NOT NULL,
            meta TEXT,
            PRIMARY KEY (collection, id)
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_col ON vectors(collection)")
    conn.commit()
    _DB = conn
    _DB_PATH = db_path
    return conn


def _norm(vec) -> np.ndarray:
    """余弦检索需要归一化向量(入库与查询统一)。返回 float32 shape (d,)。"""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v


def _match_value(meta_val, cond) -> bool:
    """对单个元数据字段值应用条件。cond 可为标量(=eq)或 op dict。"""
    if isinstance(cond, dict):
        for op, val in cond.items():
            if op == "$eq" and meta_val != val:
                return False
            if op == "$ne" and meta_val == val:
                return False
            if op == "$in" and (meta_val is None or meta_val not in val):
                return False
            if meta_val is None:
                return False
            if op == "$gt" and not (meta_val > val):
                return False
            if op == "$gte" and not (meta_val >= val):
                return False
            if op == "$lt" and not (meta_val < val):
                return False
            if op == "$lte" and not (meta_val <= val):
                return False
        return True
    if isinstance(cond, list):  # 隐式 $in
        return meta_val in cond
    return meta_val == cond


def _match_where(meta: dict, where) -> bool:
    """递归匹配 ChromaDB 风格 where: {"$and":[...]}/{"$or":[...]}/{"field":{"op":v}}。"""
    if not where:
        return True
    if not isinstance(where, dict):
        return True
    if "$and" in where:
        return all(_match_where(meta, c) for c in where["$and"])
    if "$or" in where:
        return any(_match_where(meta, c) for c in where["$or"])
    for key, cond in where.items():
        if key.startswith("$"):
            continue
        if not _match_value(meta.get(key), cond):
            return False
    return True


class _CollectionShim:
    """兼容 ChromaDB collection 的子集接口(get/delete),供 .collection 外部调用。"""

    def __init__(self, owner):
        self._owner = owner

    def get(self, include=None) -> dict:
        with self._owner._lock:
            ids = list(self._owner._items.keys())
            docs = [self._owner._items[i]["text"] for i in ids] if self._owner._items else []
            # ChromaDB 需要 documents 只在 include=[...documents] 时给,否则 None
            want_docs = include and ("documents" in include)
            return {"ids": ids, "documents": docs if want_docs else None}

    def delete(self, ids=None, where=None) -> None:
        with self._owner._lock:
            if where == {}:
                self._owner._clear_all()
                return
            if ids is not None:
                self._owner._delete_ids(list(ids))
                return
            if where:
                self._owner._delete_where(where)


class ChromaStore:
    """sqlite(持久) + faiss(内存) 确定性向量存储。"""

    def __init__(self, collection_name: str = "rag_docs", persist_dir: str = None):
        if persist_dir is None:
            persist_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "chroma_data"
            )
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._lock = threading.RLock()
        self._dim = 384
        self._items: dict[str, dict] = {}   # id -> {"vec": np.float32(d,), "text": str, "meta": dict}
        self._index: faiss.IndexFlatIP | None = None
        self._id2row: dict[str, int] = {}
        self._load_from_db()
        self.collection = _CollectionShim(self)

    # ---- 持久化 ----
    def _load_from_db(self):
        conn = _get_db(self.persist_dir)
        with self._lock:
            rows = conn.execute(
                "SELECT id, vector, text, meta FROM vectors WHERE collection=?",
                (self.collection_name,),
            ).fetchall()
            self._items = {}
            for rid, blob, text, meta in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                self._items[rid] = {
                    "vec": vec,
                    "text": text,
                    "meta": json.loads(meta) if meta else {},
                }
            self._dim = len(next(iter(self._items.values()))["vec"]) if self._items else self._dim
            self._rebuild_index()

    def _rebuild_index(self):
        ids = list(self._items.keys())
        self._id2row = {k: i for k, i in enumerate(ids)}  # 行号 -> id
        if not ids:
            self._index = faiss.IndexFlatIP(self._dim)
            return
        mat = np.vstack([self._items[i]["vec"] for i in ids])
        idx = faiss.IndexFlatIP(self._dim)
        idx.add(mat)
        self._index = idx

    def _persist(self, pairs):
        """pairs: list[(id, vec_np, text, meta_dict)] 写入 sqlite。"""
        conn = _get_db(self.persist_dir)
        conn.executemany(
            "INSERT OR REPLACE INTO vectors(collection,id,vector,text,meta) VALUES(?,?,?,?,?)",
            [(self.collection_name, rid, np.asarray(vec, np.float32).tobytes(),
              text, json.dumps(meta, ensure_ascii=False)) for rid, vec, text, meta in pairs],
        )
        conn.commit()

    def _delete_from_db(self, ids):
        conn = _get_db(self.persist_dir)
        conn.executemany(
            "DELETE FROM vectors WHERE collection=? AND id=?",
            [(self.collection_name, rid) for rid in ids],
        )
        conn.commit()

    # ---- 写 ----
    def add(self, id: str, vector, text: str, metadata: dict = None):
        self.add_batch([id], [vector], [text], [metadata or {}])

    def add_batch(self, ids, vectors, texts, metadatas=None):
        metas = metadatas or [{} for _ in ids]
        pairs = []
        for rid, v, t, m in zip(ids, vectors, texts, metas):
            nv = _norm(v)
            self._dim = len(nv)
            pairs.append((rid, nv, t, m or {}))
        with self._lock:
            self._persist(pairs)
            for rid, nv, t, m in pairs:
                self._items[rid] = {"vec": nv, "text": t, "meta": m}
            self._rebuild_index()

    def _delete_ids(self, ids):
        with self._lock:
            self._delete_from_db(ids)
            for rid in ids:
                self._items.pop(rid, None)
            self._rebuild_index()

    def _delete_where(self, where):
        with self._lock:
            rm = [rid for rid, it in self._items.items() if _match_where(it["meta"], where)]
            if rm:
                self._delete_from_db(rm)
                for rid in rm:
                    self._items.pop(rid, None)
                self._rebuild_index()

    def _clear_all(self):
        with self._lock:
            conn = _get_db(self.persist_dir)
            conn.execute("DELETE FROM vectors WHERE collection=?", (self.collection_name,))
            conn.commit()
            self._items = {}
            self._rebuild_index()

    # ---- 读 ----
    def search(self, query_vector, top_k: int = 5, where: dict = None) -> list[dict]:
        """余弦最近邻。返回 [{"id","text","similarity"}],按相似度降序,最多 top_k 条。"""
        q = _norm(query_vector)
        with self._lock:
            if not self._items or self._index is None or self._index.ntotal == 0:
                return []
            # 召回更多候选再按 where 过滤,保证过滤后仍够 top_k
            n_cand = min(max(top_k * 5, top_k + 50), self._index.ntotal)
            sims, rows = self._index.search(q.reshape(1, -1), n_cand)
            out = []
            for i, row in enumerate(rows[0]):
                if row < 0:
                    continue
                rid = self._id2row.get(int(row))
                if rid is None:
                    continue
                it = self._items[rid]
                if where and not _match_where(it["meta"], where):
                    continue
                ip = float(sims[0][i])
                # 归一化后内积 = 余弦相似度;与原 ChromaDB cosine 一致: distance=1-cos, sim=1/(1+d)
                similarity = round(1.0 / (1.0 + max(0.0, 1.0 - ip)), 4)
                out.append({"id": rid, "text": it["text"], "similarity": similarity})
                if len(out) >= top_k:
                    break
            return out

    @property
    def _data(self) -> dict:
        """兼容旧接口——返回 {id: {"text":..., "similarity":0.0}}。"""
        with self._lock:
            return {rid: {"text": it["text"], "similarity": 0.0} for rid, it in self._items.items()}

    def count(self) -> int:
        with self._lock:
            return len(self._items)
