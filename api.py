import sys, os, time, traceback, json
import asyncio, queue, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import shutil

from core.llm import LLM
from core.registry import ToolRegistry
from core.react_agent import ReactAgent
from core.workflow_agent import WorkflowAgent, RAG_WORKFLOW
from core.guard_policies import build_guard, make_ctx
from core.session_manager import SessionManager
from core.db import log_error
from core.middleware import RateLimitMiddleware, APIKeyMiddleware, RequestLogMiddleware
from tools.rag_tool import RAGTool
from tools.extra_tools import CalculatorTool, ImageAnalysisTool
from reimbursement.reimbursement_tool import ReimbursementTool
from reimbursement.service import audit_claim as reimburse_audit
from reimbursement import record_store as reimburse_records
from utils.tracer import get_tracer

# 报销票据上传根目录：固定在项目内（受 Runtime 闸门 path_root 约束，OCR 用同一套路径）
_REIMB_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reimbursement", "uploads")
os.makedirs(_REIMB_UPLOAD_DIR, exist_ok=True)
_REIMB_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pdf"}


def _guard_local_or_auth(request: Request) -> None:
    """敏感读端点守卫：已配置 API_KEY 的整体走中间件鉴权；未配置时仅放行本机访问。

    防止局域网/公网环境下，无鉴权的 /v1/reimburse/records 被遍历偷取员工报销/供应商隐私。
    本地开发(demo)完全不设 API_KEY 时，仅 127.0.0.1/localhost 可访问，不破坏开箱即用。
    """
    if os.getenv("API_KEY"):
        return  # 有鉴权中间件兜底，此处放行交由中间件校验
    client = request.client.host if request.client else ""
    if client in ("127.0.0.1", "::1", "localhost"):
        return
    raise HTTPException(status_code=403, detail="敏感数据仅限本机访问；请配置 API_KEY 后通过鉴权访问。")



class ChatRequest(BaseModel):
    message: str = Field(..., description="用户问题", min_length=1, max_length=5000)

class ChatResponse(BaseModel):
    success: bool = True
    answer: str
    trace_id: str | None = None
    steps: int = 0
    tokens: int = 0
    latency_ms: float = 0.0

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_type: str
    detail: str | None = None

class UploadResponse(BaseModel):
    success: bool = True
    message: str
    task_id: str | None = None

class UploadStatusResponse(BaseModel):
    task_id: str
    status: str          # running / success / failed
    message: str = ""
    added_chunks: int = 0

class StatsResponse(BaseModel):
    text_chunks: int
    images: int

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"



agent: ReactAgent = None
rag_tool: RAGTool = None
session_mgr: SessionManager = None

# 上传任务表 + 线程池：PDF 解析/向量化是几十秒串行任务，绝不能同步占住请求线程，
# 否则客户端等不到响应（curl 280s 超时）。改为提交后台线程、立即返回 task_id 供轮询。
_upload_tasks: dict[str, dict] = {}
from concurrent.futures import ThreadPoolExecutor
_upload_executor = ThreadPoolExecutor(max_workers=2)

# 上传任务表保留上限：任务记录只写不删会随运行时间无限增长（含解析消息，占内存）。
# 只保留最近 N 条，超出时优先裁掉已结束(success/failed)的最早记录。
_MAX_UPLOAD_TASKS = 200


def _remember_upload_task(task_id: str, info: dict) -> None:
    """登记上传任务并裁剪历史，防止 _upload_tasks 无限增长。"""
    _upload_tasks[task_id] = info
    if len(_upload_tasks) <= _MAX_UPLOAD_TASKS:
        return
    # 优先清理已结束（成功/失败）的旧记录；dict 保持插入序，从最早的开始
    for tid in list(_upload_tasks.keys()):
        if len(_upload_tasks) <= _MAX_UPLOAD_TASKS:
            break
        if _upload_tasks[tid].get("status") in ("success", "failed"):
            _upload_tasks.pop(tid, None)
    # 仍超限（都是 running）则按最早登记顺序裁剪，保证有硬上限
    while len(_upload_tasks) > _MAX_UPLOAD_TASKS:
        _upload_tasks.pop(next(iter(_upload_tasks)), None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, rag_tool, session_mgr
    try:
        llm = LLM()
        rag_tool = RAGTool()
        registry = ToolRegistry()
        registry.register(rag_tool)
        registry.register(CalculatorTool())
        registry.register(ImageAnalysisTool(llm))
        registry.register(ReimbursementTool())
        # Runtime 权限闸门：API 默认 deny（对话内高险写操作被拦截，体现"模型只出意图"），
        # 信任基线 analyst——可检索、可计算，不能写入知识库；上传走独立 /v1/upload 端点。
        registry.set_guard(build_guard(rag_tool=rag_tool, on_confirm="deny", persist=True))
        registry.set_ctx_default(make_ctx(role="analyst", user_id="api_default"))
        agent = ReactAgent(name="财务报销审核助手", llm=llm, tool_registry=registry)
        session_mgr = SessionManager(llm)
        # 预热报销问答的向量索引：启动即建好，避免"首问才建索引"卡住 20-30s（用户以为死机）
        try:
            import threading
            def _prewarm():
                try:
                    from reimbursement import rag_store as reimburse_rag
                    reimburse_rag.semantic_search("预热", top_k=1)
                    print("[启动] 报销向量索引已预热")
                except Exception as e:
                    print(f"[启动] 向量索引预热失败(将首问时兜底): {e}")
            threading.Thread(target=_prewarm, daemon=True).start()
        except Exception as e:
            print(f"[启动] 预热线程启动失败: {e}")
        print("[启动] Agent + SessionManager 已就绪")
    except Exception as e:
        print(f"[启动] 初始化失败: {e}")
        raise
    yield


app = FastAPI(title="财务报销审核平台 API", version="1.1.0", lifespan=lifespan)

app.add_middleware(RateLimitMiddleware)
if os.getenv("API_KEY"):
    app.add_middleware(APIKeyMiddleware, api_key=os.getenv("API_KEY"))
app.add_middleware(RequestLogMiddleware)

# CORS：默认仅放开本机文档站；生产请通过 CORS_ORIGINS 环境变量显式配置允许的来源。
# 注意不开启 allow_credentials（与 allow_origins=["*"] 组合是浏览器安全违规），前端用 Bearer 头鉴权即可。
_def_cors = os.getenv("CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _def_cors.split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins,
                   allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=False)



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底：所有未捕获异常返回结构化错误。客户端只拿错误类型+简讯，完整栈仅落服务端日志。"""
    error_type = type(exc).__name__
    try:
        from core.db import log_error
        log_error(error_code="UNHANDLED", error_type=error_type,
                  message=str(exc)[:300], severity="error")
    except Exception:
        pass  # 日志本身失败不阻塞响应
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error=f"服务内部错误: {str(exc)[:200]}",
            error_type=error_type,
            detail=None,
        ).model_dump(),
    )



@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse()


@app.get("/reimburse")
async def reimburse_page():
    """报销审核前端工作台（正式页面）。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "reimburse.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("frontend/reimburse.html 缺失", status_code=404)


@app.get("/demo", response_class=HTMLResponse)
async def reimburse_demo():
    """报销审核测试页：上传票据 / 手填条目 → 直接调用 /v1/reimburse 展示结论。"""
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>报销审核 · 演示</title>
<style>
body{font-family:system-ui,sans-serif;background:#f4f7fb;color:#1e293b;margin:0;padding:24px}
h1{color:#1e3a5f;font-size:20px}
.card{background:#fff;border-radius:12px;padding:24px;max-width:720px;margin:0 auto;
box-shadow:0 2px 16px rgba(30,58,95,.08);}
label{display:block;margin:14px 0 6px;font-weight:600;font-size:14px;color:#334}}
input[type=text],input[type=number],input[type=file],textarea{width:100%;padding:9px 12px;
border:1px solid #cbd5e1;border-radius:8px;font-size:14px;box-sizing:border-box}
button{background:#3182ce;color:#fff;border:none;padding:11px 22px;border-radius:8px;
font-size:15px;font-weight:600;cursor:pointer;margin-top:18px}
button:hover{background:#2c5282}
#res{margin-top:20px;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:10px;
white-space:pre-wrap;font-family:monospace;font-size:12.5px;max-height:420px;overflow:auto}
.muted{color:#64748b;font-size:12.5px}
</style></head><body>
<div class="card">
<h1>📋 报销审核演示（规则为主 + LLM 兜底）</h1>
<p class="muted">上传一张票据图，或手填报销条目，点「开始审核」。结论含：决策、可核准/拒金额、逐条核定与命中的规则。</p>
<form id="f">
<label>票据文件（png/jpg/pdf，可多个）</label>
<input type="file" id="files" name="files" multiple accept=".png,.jpg,.jpeg,.pdf">
<label>报销事由</label>
<input type="text" id="purpose" placeholder="如：北京出差住宿">
<label>申报总额（可选，用于与核定交叉核对）</label>
<input type="number" id="claimed_total" step="0.01">
<label>或：手填条目（JSON 数组，可不传文件）</label>
<textarea id="items" rows="3" placeholder='[{"category":"餐饮","date":"2026-08-01","amount":250,"desc":"晚餐商务宴请","itemization":["晚餐"]}]'></textarea>
<button type="button" onclick="run()">开始审核</button>
</form>
<div id="res">等待审核……</div>
</div>
<script>
function run(){
  const files=document.getElementById('files').files;
  const fd=new FormData();
  for(const f of files) fd.append('files', f, f.name);
  fd.append('purpose', document.getElementById('purpose').value);
  const ct=document.getElementById('claimed_total').value;
  if(ct) fd.append('claimed_total', ct);
  const items=document.getElementById('items').value.trim();
  if(items) fd.append('items_json', items);
  const res=document.getElementById('res');
  res.textContent='审核中（OCR 若首次加载会稍慢）……';
  fetch('/v1/reimburse',{method:'POST',body:fd})
    .then(r=>r.json())
    .then(j=>{
      const d=j.data;
      if(!j.ok){res.textContent='出错了：'+(j.error||JSON.stringify(j));return;}
      const v=d.verdict;
      let out='决策：'+v.decision+'\\n';
      out+='申报：¥'+v.claimed_amount+'   可核准：¥'+v.approved_amount+'   拒：¥'+v.rejected_amount+'\\n';
      out+='结论：'+v.summary+'\\n';
      if(v.items_adjudication&&v.items_adjudication.length){
        out+='\\n【逐条核定】\\n';
        v.items_adjudication.forEach(x=>{
          out+=('  '+x.decision.padEnd(13)+' ¥'+x.amount+' → ¥'+x.approved+'  ('+x.category+') '+x.desc+'\\n');
        });
      }
      if(v.issues&&v.issues.length){
        out+='\\n【命中的规则/问题】\\n';
        v.issues.forEach(i=>{out+=('  ['+i.severity+'] '+i.issue_code+' — '+i.description+'\\n');});
      }
      out+='\\n是否调用LLM兜底：'+(d.llm_used?'是':'否')+'   服务耗时：'+d.server_elapsed_ms+'ms';
      res.textContent=out;
    })
    .catch(e=>{res.textContent='请求失败：'+e;});
}
</script></body></html>"""
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
async def index():
    """根路径引导页。

    API 服务本身没有页面 UI，问答界面是独立的 Gradio(端口 7860)、WebUI 通常是
    server_port 另行部署。这里给直接访问 8000 根路径的用户一个可读的引导页，
    避免出现冷冰冰的 404。
    """
    base = os.environ.get("GRADIO_PORT", "7860")
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>智能报销审核 · API 服务</title>
<style>body{{font-family:system-ui,sans-serif;background:#f6f8fb;color:#1e293b;
display:flex;justify-content:center;align-items:center;min-height:100vh;padding:24px}}
.card{{background:#fff;border-radius:14px;padding:36px;max-width:640px;width:100%;
box-shadow:0 2px 20px rgba(30,58,95,.08);}}
h1{{font-size:22px;color:#1e3a5f;}}h1 span{{color:#3182ce;}}
p{{color:#475569;line-height:1.7;}}
code{{background:#eef2f7;padding:2px 6px;border-radius:5px;font-size:13px;}}
a.btn{{display:inline-block;margin-top:16px;background:#3182ce;color:#fff;padding:12px 24px;
border-radius:8px;text-decoration:none;font-weight:600;}}
a.btn:hover{{background:#2c5282;}}
ul{{color:#475569;}}li{{margin:6px 0;}}</style></head><body>
<div class="card"><h1>🧾 智能报销审核 <span>API 服务</span></h1>
<p>API 服务已启动。报销审核工作台与本服务同源，请访问：</p>
<a class="btn" href="/reimburse" target="_blank">打开报销审核工作台</a>
<a class="btn" style="background:#2c5282" href="/setup" target="_blank">⚙ 首次配置 LLM</a>
<h3 style="margin-top:28px">可用接口</h3>
<ul>
<li><code>GET  /health</code> 健康检查</li>
<li><code>GET  /reimburse</code> 报销审核前端（工作台）</li>
<li><code>POST /v1/reimburse</code> 报销审核接口</li>
<li><code>POST /v1/chat</code> 对话问答</li>
<li><code>POST /v1/upload</code> 文档入库</li>
<li><code>POST /v1/workflow</code> 工作流问答</li>
<li><code>POST /v1/eval</code> 离线评估</li>
<li><code>GET  /v1/stats</code> 知识库统计</li>
<li><code>GET  /setup</code> 首次配置</li>
</ul></div></body></html>"""
    return HTMLResponse(content=html)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """对话接口——会话管理 + 错误码 + 用户情景记忆。"""
    start = time.perf_counter()
    session_id = None
    tracer = get_tracer()
    tracer.start_trace("api.chat", user_message=req.message[:100])

    try:
        user_id = "default_user"
        # 从请求头提取身份与租户，供 RuntimeGuard 做角色/隔离判定（图②③）。
        # 缺省走 analyst 基线：能检索、能计算，不能写知识库。
        role = request.headers.get("X-Role", "analyst")
        tenant = request.headers.get("X-Tenant", "default")
        ctx = make_ctx(role=role, user_id=user_id, tenant_id=tenant)
        session_id = session_mgr.get_or_create(user_id)

        session_mgr.get_context(session_id, user_id)

        agent.memory = session_mgr._memories.get(session_id, agent.memory)
        result = agent.run(req.message, ctx=ctx)

        session_mgr.add_message(session_id, "user", req.message)
        session_mgr.add_message(session_id, "assistant", result)

    except ConnectionError as e:
        tracer.end_trace(success=False, error=str(e)[:200])
        log_error("NET_001", type(e).__name__, str(e), context=f"query: {req.message[:100]}")
        raise HTTPException(status_code=503, detail=f"网络连接失败: {e}")
    except TimeoutError as e:
        tracer.end_trace(success=False, error=str(e)[:200])
        log_error("NET_002", type(e).__name__, str(e), context=f"query: {req.message[:100]}")
        raise HTTPException(status_code=504, detail=f"API 响应超时: {e}")
    except Exception as e:
        error_type = type(e).__name__
        tracer.end_trace(success=False, error=str(e)[:200])
        log_error("AGENT_001", error_type, str(e), context=f"query: {req.message[:100]}")
        raise HTTPException(status_code=500, detail=f"Agent 推理异常 ({error_type}): {str(e)[:300]}")

    latency = (time.perf_counter() - start) * 1000
    trace = agent.get_last_trace()
    tracer.end_trace(success=True)

    return ChatResponse(
        success=True,
        answer=result,
        trace_id=trace.get("trace_id", "") if trace else "",
        steps=trace.get("steps", 0) if trace else 0,
        tokens=trace.get("total_tokens", 0) if trace else 0,
        latency_ms=round(latency, 1),
    )


@app.post("/v1/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """文件上传——支持 PDF/Word/Excel/PPT/HTML/EPUB/图片/纯文本。"""
    import tempfile

    ext = os.path.splitext(file.filename)[1].lower()
    max_sizes = {".pdf": 50, ".docx": 30, ".xlsx": 30, ".pptx": 50, ".epub": 50}
    max_mb = max_sizes.get(ext, 20)

    content = await file.read()
    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过限制 ({max_mb}MB)。当前格式 {ext} 最大 {max_mb}MB。"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # 安全校验：大小 + 扩展名白名单 + 文件头魔数（拒绝伪装成文档的恶意文件）
    from utils.security import validate_file
    check = validate_file(tmp_path)
    if not check["valid"]:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=check["reason"])

    # PDF 解析 + OCR + 向量化是几十秒的串行 CPU/IO 任务。绝不能同步等它——
    # 客户端会等不到响应而超时，服务端请求被占死。改成：立刻提交后台线程解析、
    # 马上返回 task_id，客户端轮询 /v1/upload/status/{task_id}。
    import uuid as _uuid
    task_id = f"up_{_uuid.uuid4().hex[:12]}"
    _remember_upload_task(task_id, {"status": "running", "message": "解析中", "added_chunks": 0})

    def _worker(tid: str, path: str):
        import json as _json
        try:
            result = rag_tool.execute("add_file", file_path=path)
            parsed = _json.loads(result)
            # setdefault：任务记录可能因上限裁剪已被移除，补建而非 KeyError
            _upload_tasks.setdefault(tid, {}).update(
                status="success" if parsed.get("ok") else "failed",
                message=result if parsed.get("ok") else parsed.get("error", "文件处理失败"),
                added_chunks=parsed.get("data", {}).get("added_chunks", 0) if parsed.get("ok") else 0,
            )
        except Exception as e:
            _upload_tasks.setdefault(tid, {}).update(status="failed", message=f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    _upload_executor.submit(_worker, task_id, tmp_path)
    return UploadResponse(success=True, task_id=task_id, message="文件已接收，正在后台解析，请轮询 /v1/upload/status/" + task_id)


@app.get("/v1/upload/status/{task_id}", response_model=UploadStatusResponse)
async def upload_status(task_id: str):
    """查询上传任务进度。status=running 时轮询，success/failed 时收尾。"""
    info = _upload_tasks.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return UploadStatusResponse(
        task_id=task_id, status=info.get("status", "running"),
        message=info.get("message", ""), added_chunks=info.get("added_chunks", 0),
    )


@app.get("/v1/stats", response_model=StatsResponse)
async def stats():
    """知识库统计"""
    try:
        return StatsResponse(
            text_chunks=rag_tool.pipeline.text_count,
            images=rag_tool.pipeline.image_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"统计查询失败: {e}")



class ReimburseResponse(BaseModel):
    ok: bool
    data: dict | None = None
    error: str = ""


async def _save_uploads(files: list[UploadFile]) -> list[str]:
    """把上传的票据保存到项目内 uploads 目录，返回落盘路径列表。

    落盘在 _REIMB_UPLOAD_DIR 下（受 path_root 约束），文件名用 uuid 防冲突，
    用户原始文件名仅做日志/展示，绝不用作落盘路径（防路径注入）。
    """
    import uuid as _uuid
    saved = []
    for f in files or []:
        ext = os.path.splitext(f.filename or "")[1].lower() or ".png"
        if ext not in _REIMB_EXT:
            ext = ".png"
        name = f"r_{_uuid.uuid4().hex[:16]}{ext}"
        dest = os.path.join(_REIMB_UPLOAD_DIR, name)
        with open(dest, "wb") as fp:
            shutil.copyfileobj(f.file, fp)
        saved.append(dest)
    return saved


@app.post("/v1/reimburse", response_model=ReimburseResponse)
async def reimburse(
    files: list[UploadFile] = File(default=[]),
    purpose: str = Form(default=""),
    claimed_total: float | None = Form(default=None),
    items_json: str = Form(default=""),
    employee_id: str = Form(default=""),
):
    """报销审核（真实使用入口）。

    两种入参（可同时）：
      - files：上传票据图/PDF，服务端 OCR + 结构化抽取 + 规则审核
      - items_json：手工录入的报销条目 JSON 列表（不走票据 OCR），用于已结构化数据
    逻辑：规则为主 + LLM 兜底（仅语义模糊点），产出一条结构化审核结论供用户执行。
    """
    import json as _json
    started = time.perf_counter()
    items = None
    if items_json:
        try:
            parsed_items = _json.loads(items_json)
            items = parsed_items if isinstance(parsed_items, list) else [parsed_items]
        except Exception:
            raise HTTPException(status_code=400, detail=f"items_json 不是合法 JSON: {items_json[:100]}")

    saved_paths: list[str] = []
    try:
        if files:
            saved_paths = await _save_uploads(files)
        result = reimburse_audit(
            items=items,
            receipt_files=saved_paths or None,
            purpose=purpose,
            claimed_total=claimed_total,
            use_llm_arbitration=True,
            employee_id=employee_id,
        )
    finally:
        for pth in saved_paths:
            try:
                os.unlink(pth)
            except OSError:
                pass

    if not result.get("ok"):
        return ReimburseResponse(ok=False, error=str(result.get("error")))

    data = result["data"]
    data["server_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)

    # 落盘本次审核（真实公司报销：关页不丢、可回看、可追问）。
    # 记录入参条目/逐条核定/命中规则/是否走LLM，供历史检索与确定性问答复用。
    try:
        record = reimburse_records.save_record(
            inputs={"purpose": purpose, "claimed_input": data["verdict"].get("claimed_input"),
                    "items": items or data["verdict"].get("items_adjudication", []),
                    "parse_errors": data["verdict"].get("parse_errors", []),
                    "employee_id": employee_id or ""},
            verdict=data["verdict"],
            llm_used=data.get("llm_used", False),
            server_elapsed_ms=data["server_elapsed_ms"],
        )
        data["record_id"] = record["id"]
    except Exception as e:
        data["record_id"] = None
        data["record_warn"] = f"记录落盘失败: {e}"
    return ReimburseResponse(ok=True, data=data)


# ---- 审核记录：历史检索 / 单笔详情 / 确定性问答（真实公司报销系统能力）----

class ReimburseRecordListResponse(BaseModel):
    ok: bool = True
    records: list[dict] = Field(default_factory=list)
    total: int = 0
    error: str = ""


class ReimburseAskRequest(BaseModel):
    record_id: str = Field(..., description="审核记录 id")
    question: str = Field(..., description="对这笔审核的提问", min_length=1, max_length=500)


class ReimburseAskResponse(BaseModel):
    ok: bool
    answer: str = ""
    intent: list[str] = Field(default_factory=list)
    record_id: str = ""
    error: str = ""


class ReimburseSysAskRequest(BaseModel):
    question: str = Field(..., description="对整库的提问(统计/查询/计算)", min_length=1, max_length=500)
    context: dict = Field(default_factory=dict, description="上一轮实体，多轮对话承接")


class ReimburseSysAskResponse(BaseModel):
    ok: bool
    answer: str = ""
    intent: list[str] = Field(default_factory=list)
    records: list[dict] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)
    llm_used: bool = False
    error: str = ""


@app.get("/v1/reimburse/records", response_model=ReimburseRecordListResponse)
async def reimburse_record_list(request: Request, keyword: str = "", decision: str = "",
                                limit: int = 50, employee_id: str = ""):
    """历史审核记录（新在前）。支持 关键词/决策档位/员工工号 过滤。含员工/供应商隐私，默认本机保护。"""
    _guard_local_or_auth(request)
    try:
        recs = reimburse_records.list_records(keyword=keyword, decision=decision,
                                              limit=max(1, min(limit, 200)),
                                              employee_id=employee_id)
        return ReimburseRecordListResponse(ok=True, records=recs, total=len(recs))
    except Exception as e:
        return ReimburseRecordListResponse(ok=False, records=[], total=0, error=str(e))


@app.get("/v1/reimburse/records/{rec_id}", response_model=ReimburseRecordListResponse)
async def reimburse_record_detail(request: Request, rec_id: str):
    """单笔审核完整详情（含逐条核定 / 命中规则 / 是否走LLM）。含员工/供应商隐私，默认本机保护。"""
    _guard_local_or_auth(request)
    try:
        rec = reimburse_records.get_record(rec_id)
        if not rec:
            return ReimburseRecordListResponse(ok=False, records=[], total=0, error=f"记录 {rec_id} 不存在")
        return ReimburseRecordListResponse(ok=True, records=[rec], total=1)
    except Exception as e:
        return ReimburseRecordListResponse(ok=False, records=[], total=0, error=str(e))


@app.post("/v1/reimburse/ask", response_model=ReimburseAskResponse)
async def reimburse_ask(req: ReimburseAskRequest):
    """对单笔审核做**确定性**问答（读该笔核定记录 + 规则直接回答，0 LLM、可验证）。"""
    try:
        rec = reimburse_records.get_record(req.record_id)
        if not rec:
            return ReimburseAskResponse(ok=False, record_id=req.record_id,
                                        error=f"记录 {req.record_id} 不存在")
        ans = reimburse_records.answer_question(rec, req.question)
        return ReimburseAskResponse(ok=True, answer=ans["answer"], intent=ans.get("intent", []),
                                    record_id=req.record_id)
    except Exception as e:
        return ReimburseAskResponse(ok=False, record_id=req.record_id, error=str(e))


# 语义召回来源的相似度阈值：低于该值的记录视为"不够相关"，不作为来源展示(像豆包只陈列有关的)
_SIM_THRESHOLD = 0.30


def _is_semantic(q: str) -> bool:
    """这类问法需要组织性/解释性回答 → 应走 RAG 生成(检索 + LLM 依据事实组织)。"""
    return any(
        k in q for k in ["为什么", "为啥", "怎么", "如何", "哪些", "哪些单", "有什么", "解释",
                         "原因", "情况", "说明", "区别", "谁", "哪笔", "怎么算", "怎么看", "怎么样",
                         "总结", "概括", "点评", "分析", "到底", "究竟", "是否", "合理", "对吗",
                         "可疑", "有问题", "异常", "不对劲", "为何"])


def _facts_for_rag(records: list[dict], stats: dict, label: str) -> str:
    """把检索到的记录组织成"只含事实"的文本，供 LLM 生成时引用(禁止编造)。"""
    dec_cn = {"approve": "通过", "partial": "部分核准", "reject": "不予报销", "manual_review": "待人工复核"}
    lines = []
    lines.append(f"【检索范围】{label}")
    lines.append(
        f"【合计】命中 {stats.get('n', len(records))} 笔：申报 ¥{stats.get('claimed', 0):,.2f}、"
        f"核准 ¥{stats.get('approved', 0):,.2f}、拒付 ¥{stats.get('rejected', 0):,.2f}。")
    for r in records:
        dec = dec_cn.get(r.get("decision"), r.get("decision", ""))
        lines.append(
            f"- {str(r.get('ts', ''))[:10]} 「{r.get('purpose') or '（未填事由）'}」 [{dec}]："
            f"申报 ¥{r.get('claimed_amount', 0):,.2f} → 核准 ¥{r.get('approved_amount', 0):,.2f}，"
            f"拒 ¥{r.get('rejected_amount', 0):,.2f}，命中 {r.get('issue_count', 0)} 项规则。")
        for it in r.get("items_adjudication", []):
            lines.append(
                f"    · 条目「{it.get('desc', '')}」类别 {it.get('category', '')}："
                f"¥{it.get('amount', 0):,.2f} → ¥{it.get('approved', 0):,.2f}（{dec_cn.get(it.get('decision'), it.get('decision', ''))}）")
        for iss in r.get("issues", []):
            lines.append(f"    · 命中规则 {iss.get('issue_code', '')}：{iss.get('description', '')}")
    return "\n".join(lines)


@app.post("/v1/reimburse/ask_system", response_model=ReimburseSysAskResponse)
async def reimburse_ask_system(req: ReimburseSysAskRequest):
    """**向量语义 RAG** 整库问答：语义召回 + 结构化锚定 + LLM 生成。

    - 检索：① 向量语义检索 `reimburse/rag_store.semantic_search` —— 问题 embedding，
      与已入库的审核记录做余弦召回，**口语化/模糊表述也能召回相关记录**；
      ② 结构化聚合(决策/月份/类别/金额)负责**数字锚定**——多少笔/总额/核减由规则过滤给出，保证准。
    - 生成：语义类问题(为什么/哪些/怎么样/可疑)由 LLM **依据召回的事实**组织自然语言回答，
      prompt 强约束"只能引用给定事实、禁止编造数字"，命中明细可展开核验。
    - 纯数字统计类(累计多少/多少笔)仍走确定性模板：准且快，不耗模型。
    """
    try:
        ans = reimburse_records.answer_system(req.question, context=req.context or None)
        llm_used = False

        # 向量语义召回：为口语/解释类问题补足"相关的记录"
        sem_records = []
        try:
            from reimbursement import rag_store as reimburse_rag
            sem_records = ([x["record"] for x in reimburse_rag.semantic_search(req.question, top_k=8)
                            if x.get("similarity", 0) >= _SIM_THRESHOLD])[:3]
        except Exception:
            sem_records = []

        # 供展开核验的记录：语义/解释类用语义召回(更贴话语)，纯数字统计用结构化命中(完整)
        struct_records = ans.get("records", [])
        base_records = struct_records
        usable_rag = False          # 是否该走 LLM 依据事实生成
        if sem_records and _is_semantic(req.question):
            base_records = sem_records      # 语义/解释类 → 语义召回(更贴话语)
            usable_rag = True
        elif not struct_records and sem_records:
            base_records = sem_records      # 兜底：结构化没命中但语义召回有 → 不再报"没有符合记录"
            usable_rag = True
        elif struct_records and _is_semantic(req.question):
            usable_rag = True               # 命中且语义类 → RAG 生成
        # 其余(命中且纯数字统计类)保持确定性：准且快、0 LLM

        if base_records and usable_rag:
            ctx = ans.get("context") or {}
            label = " · ".join(str(x) for x in [
                _ent_zh(ctx.get("decision")) if ctx.get("decision") else "",
                ctx.get("month") or "",
                ctx.get("category") or "",
                f"≥{ctx['money_gte']:g}元" if ctx.get("money_gte") is not None else "",
            ] if x) or "全库"
            sys_prompt = (
                    "你是报销审核系统的问答助手。下面是系统从审核记录库【检索到的事实】。"
                    "请**只依据这些事实**回答用户问题：数字、笔数、决策、规则只能来自事实，"
                    "禁止编造事实里没有的金额或结论；用简体中文、口语自然、有条理；"
                    "若事实不足以回答，说明缺什么并给出可以进一步查询的方向。"
                    "输出用**简洁、分段**的文字：结论先行、按要点分点说明、金额笔数加粗、要点之间换行；"
                    "若你判断该问题与检索到的审核记录无关（如天气/娱乐等非报销话题），"
                    "请在答案第一行先输出【无关】二字，再简短说明。")
            user_prompt = (f"用户问题：{req.question}\n\n检索到的事实：\n"
                           + _facts_for_rag(base_records, ans.get("stats") or {}, label))
            try:
                from core.llm import LLM
                text = LLM().chat(
                    [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user_prompt}], temperature=0.3)
                if text and text.strip():
                    ans["answer"] = text.strip()
                    llm_used = True
            except Exception as e:
                # LLM 失败 → 保留确定性回答(附注说明)
                ans["answer"] = (ans["answer"] or "") + f"\n\n（注：语言组织服务暂不可用，此为检索摘要。）"
        return ReimburseSysAskResponse(ok=True, answer=ans["answer"], intent=ans.get("intent", []),
                                       records=base_records, stats=ans.get("stats", {}),
                                       context=ans.get("context", {}), llm_used=llm_used)
    except Exception as e:
        return ReimburseSysAskResponse(ok=False, error=str(e))


async def _aiter_sync(sync_iter, should_stop=None):
    """把同步生成器放到后台线程拉取、经队列喂给事件循环——避免阻塞 async 端点。

    LLM 的 chat_stream 是同步阻塞生成器；在 async 端点里直接 `for` 迭代会占住
    事件循环，期间其他并发请求全被拖住。这里用「后台线程 + 无界队列」解耦：
    线程负责阻塞式拉取，事件循环只 await 队列取数，互不阻塞。
    should_stop：可选回调，返回 True 时线程停止继续拉取（客户端断开时止损）。
    """
    q: "queue.Queue" = queue.Queue()
    _SENTINEL = object()

    def _pump():
        try:
            for item in sync_iter:
                if should_stop is not None and should_stop():
                    break
                q.put(item)
        except BaseException as exc:      # 异常也交给消费方按原语义抛出
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=_pump, daemon=True).start()

    get_task = asyncio.ensure_future(asyncio.to_thread(q.get))
    try:
        while True:
            item = await get_task
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            get_task = asyncio.ensure_future(asyncio.to_thread(q.get))  # 预取下一项
            yield item
    finally:
        get_task.cancel()


@app.post("/v1/reimburse/ask_system/stream")
async def reimburse_ask_system_stream(req: ReimburseSysAskRequest, request: Request):
    """**流式 RAG 问答**：检索(确定性/语义)同 ask_system，但 LLM 生成逐块吐出(打字机)。

    响应为 NDJSON，每行一个对象：
      {"type":"meta", records, stats, context, intent, llm_used}  一次性元数据(前端先渲染来源)
      {"type":"delta","text":str}   生成增量(可多次)
      {"type":"done","answer":str}  结束
      {"type":"error","message":str} 失败
    前端用 fetch+ReadableStream 逐行解析，可 AbortController 中途停止。

    客户端中途断开：每个增量前检查 request.is_disconnected()，断开即停止继续生成 LLM，
    避免用户已经离开、后端还在把整段答案烧完。
    """

    def _j(o: dict) -> str:
        return json.dumps(o, ensure_ascii=False, default=str) + "\n"

    async def _client_gone() -> bool:
        try:
            return await request.is_disconnected()
        except Exception:
            return False

    async def gen():
        if await _client_gone():
            return
        try:
            # 确定性检索也可能耗时（SQL/聚合），放线程执行，别占住事件循环
            ans = await asyncio.to_thread(
                reimburse_records.answer_system, req.question,
                context=req.context or None)
        except Exception as e:
            yield _j({"type": "error", "message": f"检索失败：{e}"})
            return
        sem_records = []
        try:
            from reimbursement import rag_store as reimburse_rag
            # 语义检索含 embedding 计算，是重 CPU/IO 操作，同样放线程
            sem_records = await asyncio.to_thread(
                lambda: ([x["record"] for x in reimburse_rag.semantic_search(req.question, top_k=8)
                          if x.get("similarity", 0) >= _SIM_THRESHOLD])[:3])
        except Exception:
            sem_records = []

        struct_records = ans.get("records", [])
        base_records = struct_records
        usable_rag = False
        if sem_records and _is_semantic(req.question):
            base_records = sem_records; usable_rag = True
        elif not struct_records and sem_records:
            base_records = sem_records; usable_rag = True
        elif struct_records and _is_semantic(req.question):
            usable_rag = True

        ctx = ans.get("context") or {}
        label = " · ".join(str(x) for x in [
            _ent_zh(ctx.get("decision")) if ctx.get("decision") else "",
            ctx.get("month") or "",
            ctx.get("category") or "",
            f"≥{ctx['money_gte']:g}元" if ctx.get("money_gte") is not None else "",
        ] if x) or "全库"

        # 先发元数据：来源记录/统计/context，前端可立即渲染来源区，文本随后流式进来
        yield _j({"type": "meta", "records": base_records, "stats": ans.get("stats", {}),
                  "context": ctx, "intent": ans.get("intent", []), "llm_used": usable_rag,
                  "answer_prefix": "" if usable_rag else str(ans.get("answer", ""))})

        if base_records and usable_rag:
            sys_prompt = (
                "你是报销审核系统的问答助手。下面是系统从审核记录库【检索到的事实】。"
                "请**只依据这些事实**回答用户问题：数字、笔数、决策、规则只能来自事实，"
                "禁止编造事实里没有的金额或结论；用简体中文、口语自然、有条理；"
                "若事实不足以回答，说明缺什么并给出可以进一步查询的方向。"
                "输出用**简洁、分段**的文字：结论先行、按要点分点说明、金额笔数加粗、要点之间换行；"
                "若你判断该问题与检索到的审核记录无关（如天气/娱乐等非报销话题），"
                "请在答案第一行先输出【无关】二字，再简短说明。")
            user_prompt = (f"用户问题：{req.question}\n\n检索到的事实：\n"
                           + _facts_for_rag(base_records, ans.get("stats") or {}, label))
            try:
                from core.llm import LLM
                parts = []
                gone = False
                # 同步 chat_stream 经 _aiter_sync 放后台线程拉取，不阻塞事件循环；
                # 每个增量前检查客户端是否断开，断开即止损、不再继续消耗 LLM。
                stream = LLM().chat_stream(
                    [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user_prompt}], temperature=0.3)
                async for ch in _aiter_sync(stream, should_stop=lambda: gone):
                    if await _client_gone():
                        gone = True
                        return
                    parts.append(ch)
                    yield _j({"type": "delta", "text": ch})
                ans["answer"] = "".join(parts)
            except Exception as e:
                # LLM 失败 → 补一句说明，仍给出检索摘要
                note = "\n\n（注：语言组织服务暂不可用，此为检索摘要。）"
                yield _j({"type": "delta", "text": note})
                ans["answer"] = (ans["answer"] or "") + note
        else:
            # 纯确定性统计类：不流式，直接把完整 answer 一次性给
            yield _j({"type": "delta", "text": str(ans.get("answer", ""))})

        # 无关问题：剔除【无关】标记并清空来源记录→前端不再显示来源区(像豆包"没有就不显示")
        final_ans = str(ans.get("answer", ""))
        final_records = base_records
        if "【无关】" in final_ans:
            final_ans = final_ans.replace("【无关】", "").strip()
            final_records = []
        yield _j({"type": "done", "answer": final_ans, "llm_used": usable_rag, "records": final_records})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _ent_zh(d: str) -> str:
    return {"reject": "拒报", "approve": "通过", "partial": "部分核准",
            "manual_review": "待人工复核"}.get(d, d or "")


class EvalRequest(BaseModel):
    question: str = Field(..., description="测试问题")
    answer: str = Field(..., description="Agent 生成的答案")
    contexts: list[str] = Field(default_factory=list, description="检索到的上下文块（可选，不传则重新检索）")

class EvalResponse(BaseModel):
    success: bool = True
    question: str
    passed: bool
    scores: dict
    hallucination: dict
    summary: str


@app.post("/v1/eval", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    """RAG 质量评估——忠实度 + 答案相关性 + 上下文精度 + 幻觉检测。

    用于回归测试：改完检索策略后跑一次评估，确认质量不降。
    """
    from eval.rag_evaluator import RAGEvaluator, HallucinationDetector

    try:
        contexts = req.contexts
        if not contexts:
            raw_results = rag_tool.pipeline.search(req.question, top_k=5)
            contexts = [r["text"] for r in raw_results]

        evaluator = RAGEvaluator(agent.llm)
        scores = evaluator.evaluate(req.question, req.answer, contexts)

        hd = HallucinationDetector(agent.llm)
        hallucination = hd.detect(req.answer, contexts)

        passed = (
            scores["faithfulness"]["score"] >= 0.5
            and scores["answer_relevancy"]["score"] >= 0.5
            and scores["context_precision"]["score"] >= 0.3
            and hallucination["hallucination_risk"] != "高"
        )

        summary_parts = []
        if scores["faithfulness"]["score"] < 0.5:
            summary_parts.append(f"忠实度低({scores['faithfulness']['score']})→可能幻觉")
        if scores["answer_relevancy"]["score"] < 0.5:
            summary_parts.append(f"相关性低({scores['answer_relevancy']['score']})→偏题")
        if scores["context_precision"]["score"] < 0.3:
            summary_parts.append(f"检索精度低({scores['context_precision']['score']})→需优化")
        if hallucination["hallucination_risk"] == "高":
            summary_parts.append("幻觉风险高→需检查答案真实性")

        return EvalResponse(
            success=True,
            question=req.question[:100],
            passed=passed,
            scores=scores,
            hallucination=hallucination,
            summary="；".join(summary_parts) if summary_parts else "所有指标通过",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {e}")



@app.post("/v1/workflow", response_model=ChatResponse)
async def workflow_chat(req: ChatRequest):
    """工作流模式——用预设 RAG 流水线执行，适合结构化检索任务。"""
    start = time.perf_counter()
    tracer = get_tracer()
    tracer.start_trace("api.workflow", user_message=req.message[:100])

    try:
        wf_agent = WorkflowAgent(
            name="财务报销审核工作流",
            llm=agent.llm,
            tool_registry=agent.tool_registry,
            workflow=RAG_WORKFLOW,
        )
        result = wf_agent.run(req.message)
        tracer.end_trace(success=True)
    except Exception as e:
        tracer.end_trace(success=False, error=str(e)[:200])
        raise HTTPException(status_code=500, detail=f"Workflow 执行失败: {e}")

    latency = (time.perf_counter() - start) * 1000
    return ChatResponse(
        success=True,
        answer=result,
        trace_id="",
        steps=len(RAG_WORKFLOW),
        latency_ms=round(latency, 1),
    )


class SetupRequest(BaseModel):
    LLM_BASE_URL: str = ""; LLM_API_KEY: str = ""; LLM_TEXT_MODEL: str = ""
    LLM_IMAGE_MODEL: str = ""

@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    path = os.path.join(os.path.dirname(__file__), "setup.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>setup.html not found</h1>"

@app.post("/setup")
async def setup_save(req: SetupRequest):
    """写入/更新 LLM 配置到 .env，并热更新当前进程配置（无需重启）。

    安全限制：
      - base_url 必须是 https（防中间人把请求导向恶意端点窃取内容）。
      - 已配置过 key 时，不允许覆盖为空或覆盖为新的非空 key（防止远程把 key 改成
        攻击者自己的、或清空）；只能新增（空→有）。首次配置/重置需人工清 .env。
    """
    data = req.model_dump()
    base = (data.get("LLM_BASE_URL") or "").strip()
    if base and not base.lower().startswith("https://"):
        raise HTTPException(status_code=400, detail="API Base URL 必须为 https 端点。")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    # 读取现存配置，用于"已配置则不允许覆盖/清空"
    existing = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    except FileNotFoundError:
        pass

    new_key = (data.get("LLM_API_KEY") or "").strip()
    if existing.get("LLM_API_KEY") and new_key and new_key != existing["LLM_API_KEY"]:
        # 已有 key 且请求要改成不同 key：拒绝（避免被远程替换）。重置请手动清 .env。
        raise HTTPException(status_code=403,
                            detail="已配置 LLM_API_KEY，为安全起见不允许在线替换；如需更换请手动编辑 .env。")
    if existing.get("LLM_API_KEY") and not new_key:
        # 请求试图清空 key：同样拒绝
        raise HTTPException(status_code=403, detail="不允许在线清空已有 LLM_API_KEY。")

    # 合并：保留 Comment/其他已有配置行，仅更新 LLM_* 四键
    merged = {k: v for k, v in existing.items() if k not in
              ("LLM_BASE_URL", "LLM_API_KEY", "LLM_TEXT_MODEL", "LLM_IMAGE_MODEL")}
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_TEXT_MODEL", "LLM_IMAGE_MODEL"):
        v = data.get(k) or ""
        if v:
            merged[k] = v.strip()

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("# 多模态 RAG 配置 — 由 /setup 或设置入口维护，请勿提交含真实密钥的 .env 到版本库\n")
        for k, v in merged.items():
            f.write(f"{k}={v}\n")

    try:
        from core.llm import reload_config
        reload_config()  # 热更新，当前进程立即生效
    except Exception:
        pass
    return {"ok": True, "configured": bool(merged.get("LLM_API_KEY"))}


if __name__ == "__main__":
    import uvicorn
    # 端口/主机位支持从环境读取（launcher 已加载 .env 并透传）；默认维持 8000
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")))
