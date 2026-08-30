import time, json, os
from datetime import datetime


class TraceLogger:
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._current_trace = None

    def start_trace(self, user_input: str, agent_name: str = "") -> str:
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(user_input) % 10000:04d}"
        self._current_trace = {
            "trace_id": trace_id, "agent": agent_name, "user_input": user_input,
            "start_time": time.time(), "total_tokens": 0, "steps": [],
            "final_answer": None, "success": None, "error": None,
        }
        return trace_id

    def log_step(self, step_num: int, thought: str, action: str,
                 observation: str, tokens: int = 0, latency_ms: float = 0.0, success: bool = True):
        if self._current_trace is None: return
        self._current_trace["steps"].append({
            "step": step_num, "thought": thought[:200], "action": action,
            "observation": observation[:300], "tokens": tokens,
            "latency_ms": round(latency_ms, 1), "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        self._current_trace["total_tokens"] += tokens

    def end_trace(self, final_answer: str = "", success: bool = True, error: str = ""):
        if self._current_trace is None: return
        self._current_trace["final_answer"] = final_answer[:500]
        self._current_trace["success"] = success
        self._current_trace["error"] = error
        self._current_trace["total_time_ms"] = round((time.time() - self._current_trace["start_time"]) * 1000, 1)
        trace = self._current_trace
        self._current_trace = None
        self._write_trace(trace)
        self._print_summary(trace)

    def list_traces(self, limit: int = 20) -> list[dict]:
        files = sorted([f for f in os.listdir(self.log_dir) if f.endswith(".json")], reverse=True)[:limit]
        traces = []
        for f in files:
            try:
                with open(os.path.join(self.log_dir, f), "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    traces.append({
                        "trace_id": data["trace_id"], "user_input": data["user_input"][:60],
                        "steps": len(data["steps"]), "total_tokens": data["total_tokens"],
                        "success": data["success"], "time_ms": data.get("total_time_ms", 0),
                    })
            except Exception: continue
        return traces

    def _write_trace(self, trace: dict):
        filepath = os.path.join(self.log_dir, f"{trace['trace_id']}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

    def _print_summary(self, trace: dict):
        steps = len(trace["steps"])
        tokens = trace["total_tokens"]
        time_ms = trace.get("total_time_ms", 0)
        status = "✅" if trace["success"] else "❌"
        print(f"\n{'─'*50}\n  Trace: {trace['trace_id']}  {status}")
        print(f"  输入: {trace['user_input'][:60]}\n  步数: {steps} | Token: {tokens} | 耗时: {time_ms:.0f}ms")
        for step in trace["steps"]:
            print(f"    Step{step['step']} {'✅' if step['success'] else '❌'} {step['action'][:50]}")
        print(f"{'─'*50}\n")
