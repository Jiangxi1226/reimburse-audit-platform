import time, json, os, sys, threading
from datetime import datetime
from contextlib import contextmanager

# Windows 控制台默认 GBK，无法输出 ✓/✗/└─ 等 Unicode 符号，导致 _print_tree 的 print()
# 抛 UnicodeEncodeError，把正常 Agent 结果吞成 500。强制 stdout 用 UTF-8 且容错替换。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError, OSError):
    pass



class Span:
    """一个追踪单元——有父有子、有起止时间。"""
    __slots__ = (
        "span_id", "parent_id", "trace_id", "name",
        "start_time", "end_time", "duration_ms",
        "attributes", "status", "events", "children",
    )

    def __init__(self, name: str, trace_id: str = None,
                 parent_id: str = None):
        import uuid
        self.span_id = str(uuid.uuid4())[:16]
        self.parent_id = parent_id
        self.trace_id = trace_id or self.span_id
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time = None
        self.duration_ms = 0
        self.attributes: dict = {}
        self.status: str = "running"
        self.events: list[dict] = []
        self.children: list[Span] = []

    def set_attribute(self, key: str, value):
        """设置元数据——模型名、token数、相似度阈值等。"""
        self.attributes[key] = value

    def add_event(self, name: str, **attrs):
        """添加事件——'检索命中5条'、'Reranker精排完成'等。"""
        self.events.append({
            "name": name, "timestamp": time.perf_counter(),
            "attributes": attrs,
        })

    def set_status(self, status: str):
        self.status = status

    def finish(self):
        """结束 span——记录结束时间和耗时。"""
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        if self.status == "running":
            self.status = "ok"

    def to_dict(self) -> dict:
        """递归序列化——包括所有子 span。"""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": datetime.now().isoformat(),
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "events": self.events,
            "children": [c.to_dict() for c in self.children],
        }



class Tracer:
    """链路追踪器——管理 Trace 生命周期。

    每个线程独立 trace（threading.local），保证并发安全。
    """

    def __init__(self, log_dir: str = None, enabled: bool = True):
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "traces"
            )
        self.log_dir = log_dir
        self.enabled = enabled
        self._local = threading.local()

        if enabled:
            os.makedirs(self.log_dir, exist_ok=True)


    def start_trace(self, name: str, **attrs) -> Span:
        """开始一次追踪——创建根 span。"""
        if not self.enabled:
            return Span(name)

        span = Span(name)
        for k, v in attrs.items():
            span.set_attribute(k, v)

        if not hasattr(self._local, "current_spans"):
            self._local.current_spans = []
        self._local.current_spans.append(span)
        self._local.root_span = span

        return span

    def end_trace(self, success: bool = True, error: str = ""):
        """结束当前 trace——完成根 span，写 JSON。"""
        if not self.enabled:
            return

        root = getattr(self._local, "root_span", None)
        if root is None:
            return

        root.finish()
        root.set_status("ok" if success else "error")
        if error:
            root.add_event("error", message=error)

        self._write(root)
        self._print_tree(root)

        self._local.current_spans = []
        self._local.root_span = None


    @contextmanager
    def start_span(self, name: str, **attrs):
        """上下文管理器——自动开始/结束子 span。

        用法：
          with tracer.start_span("chromadb.query", top_k=5) as span:
              results = store.search(...)
              span.set_attribute("hits", len(results))
        """
        if not self.enabled or not hasattr(self._local, "current_spans"):
            yield Span(name)
            return

        parent = self._local.current_spans[-1] if self._local.current_spans else None
        span = Span(name, trace_id=parent.trace_id if parent else None,
                   parent_id=parent.span_id if parent else None)

        if parent:
            parent.children.append(span)
        self._local.current_spans.append(span)

        try:
            yield span
        except Exception as e:
            span.set_status("error")
            span.add_event("exception", message=str(e)[:200])
            raise
        finally:
            span.finish()
            self._local.current_spans.pop()


    def _write(self, root: Span):
        """写 JSON 文件。"""
        filepath = os.path.join(self.log_dir, f"{root.trace_id}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(root.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _print_tree(self, span: Span, indent: int = 0):
        """终端打印度树——快速排查。"""
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        status = "✓" if span.status == "ok" else "✗"
        attrs = ""
        if span.attributes:
            key_attrs = {k: v for k, v in list(span.attributes.items())[:3]}
            attrs = f" [{', '.join(f'{k}={v}' for k, v in key_attrs.items())}]"
        print(f"{prefix}{status} {span.name} ({span.duration_ms}ms){attrs}")
        for child in span.children:
            self._print_tree(child, indent + 1)


    def find_slow_spans(self, trace_id: str = None, min_ms: float = 500) -> list[dict]:
        """查找耗时超过 min_ms 的 span。用于性能优化。"""
        if trace_id:
            filepath = os.path.join(self.log_dir, f"{trace_id}.json")
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    root = json.load(f)
                return self._find_slow(root, min_ms)
            return []

        slow = []
        for f in os.listdir(self.log_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(self.log_dir, f), "r", encoding="utf-8") as fp:
                        root = json.load(fp)
                    slow.extend(self._find_slow(root, min_ms))
                except Exception:
                    pass
        return sorted(slow, key=lambda x: x["duration_ms"], reverse=True)

    def _find_slow(self, node: dict, min_ms: float) -> list[dict]:
        """递归查找慢 span。"""
        results = []
        if node.get("duration_ms", 0) >= min_ms:
            results.append({
                "name": node["name"],
                "duration_ms": node["duration_ms"],
                "span_id": node["span_id"],
                "trace_id": node["trace_id"],
                "attributes": node.get("attributes", {}),
            })
        for child in node.get("children", []):
            results.extend(self._find_slow(child, min_ms))
        return results



_tracer_instance: Tracer | None = None

def get_tracer() -> Tracer:
    """获取全局 Tracer 单例。"""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = Tracer()
    return _tracer_instance
