import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from core.llm import LLM
from core.registry import ToolRegistry
from core.react_agent import ReactAgent
from core.workflow_agent import WorkflowAgent, RAG_WORKFLOW
from core.guard_policies import build_guard, make_ctx
from tools.rag_tool import RAGTool
from tools.extra_tools import CalculatorTool, ImageAnalysisTool
from core.chat_store import (
    list_sessions, create_session, get_messages,
    save_exchange, delete_session,
)

NEW_SESSION = "__new__"  # 下拉框中"新建会话"占位值

llm = LLM()
registry = ToolRegistry()
rag_tool = RAGTool()
registry.register(rag_tool)
registry.register(CalculatorTool())
registry.register(ImageAnalysisTool(llm))
# Runtime 权限闸门：本地 Gradio 演示按 admin 全量信任（读/写/确认均放行）。
# 工具执行权统一交给 registry 的 Guard，模型产物仅视为意图。
registry.set_guard(build_guard(rag_tool=rag_tool, on_confirm="approve", persist=True))
registry.set_ctx_default(make_ctx(role="admin", user_id="local_demo"))

agent = ReactAgent(name="财务报销审核助手", llm=llm, tool_registry=registry)
workflow_agent = WorkflowAgent(
    name="财务报销审核工作流", llm=llm, tool_registry=registry,
    workflow=RAG_WORKFLOW,
)

BUSINESS_CSS = """
:root {
    /* 中性灰墨主色：极简商务，只保留成功/警告/危险三个语义色 */
    --primary: #20242b;
    --primary-light: #2f353d;
    --accent: #20242b;
    --success: #16a34a;
    --warning: #b7791f;
    --danger: #c53030;
    --bg: #fafbfc;
    --card: #ffffff;
    --text: #1a202c;
    --text-secondary: #69727d;
    --border: #e6e9ed;
    --radius: 10px;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
    background: var(--bg) !important;
}

/* 头部导航 */
.header {
    background: var(--card);
    color: var(--text);
    padding: 22px 28px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header h1 { margin: 0; font-size: 21px; font-weight: 600; letter-spacing: 0.3px; }
.header .subtitle { font-size: 13px; color: var(--text-secondary); margin-top: 4px; }
.header .status-dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--success);
    display: inline-block; margin-right: 6px;
}
.header .status-text { font-size: 12px; color: var(--text-secondary); }

/* 卡片 */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* 标签页 */
.tabs { border: none !important; }
.tab-nav { background: var(--card) !important; border-radius: var(--radius) !important; padding: 4px !important; }
.tab-nav button {
    border-radius: 6px !important; padding: 8px 20px !important;
    font-weight: 500 !important; font-size: 14px !important;
}
.tab-nav button.selected { background: var(--accent) !important; color: white !important; }

/* 输入框 */
input, textarea, .file-preview {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    transition: border-color 0.2s !important;
}
input:focus, textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(32,36,43,0.07) !important; }

/* 按钮 */
button.primary {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: var(--radius) !important;
    padding: 10px 24px !important; font-weight: 500 !important;
    transition: all 0.2s !important;
}
button.primary:hover { background: var(--primary) !important; }

/* 聊天区域 */
.chatbot { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; }

/* 错误 Toast */
.toast-error {
    background: var(--danger);
    color: white;
    padding: 12px 16px !important; font-size: 13px !important;
}
.toast-warning {
    background: var(--warning);
    color: white;
    padding: 12px 16px !important; font-size: 13px !important;
}
.toast-success {
    background: var(--success);
    color: white;
    padding: 12px 16px !important; font-size: 13px !important;
}

/* 统计卡片 */
.stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; text-align: center;
}
.stat-card .stat-value { font-size: 28px; font-weight: 700; color: var(--primary); }
.stat-card .stat-label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

footer { text-align: center; color: var(--text-secondary); font-size: 12px; padding: 20px; }

/* 细腻滚动条 */
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 8px; }
*::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
*::-webkit-scrollbar-track { background: transparent; }

/* 卡片微交互(轻浮起,不喧宾夺主) */
.card, .stat-card {
    transition: box-shadow .2s ease, transform .2s ease;
}
.card:hover, .stat-card:hover {
    box-shadow: 0 5px 14px rgba(18,58,99,.08);
    transform: translateY(-1px);
}

/* 按钮统一圆角与悬停过渡 */
button {
    border-radius: var(--radius) !important;
}
"""


def _safe_agent_run(message: str) -> tuple[str, str, str]:
    """安全执行 Agent 推理，返回 (答案, 状态标签, 跟踪摘要)。

    支持命令前缀：
      /workflow → 使用 WorkflowAgent（结构化流水线）
      /hybrid   → 在检索时使用混合检索（BM25+向量）
    """
    if not message or not message.strip():
        return "", "", ""

    use_workflow = message.startswith("/workflow ")
    use_hybrid = message.startswith("/hybrid ")
    clean_message = message
    if use_workflow:
        clean_message = message[10:]
    elif use_hybrid:
        clean_message = message[8:]

    try:
        start = time.perf_counter()

        if use_workflow:
            result = workflow_agent.run(clean_message)
            steps = len(RAG_WORKFLOW)
        else:
            result = agent.run(clean_message)
            trace = agent.get_last_trace()
            steps = trace.get("steps", "?") if trace else "?"

        latency = (time.perf_counter() - start) * 1000
        mode_tag = "[Workflow]" if use_workflow else ("[Hybrid]" if use_hybrid else "")
        status = f"✓ 完成{mode_tag} · {steps} 步 · {latency:.0f}ms"

        trace = agent.get_last_trace() if not use_workflow else None
        trace_summary = _format_trace(trace) if trace else ""
    except Exception as e:
        error_msg = str(e)[:300]
        result = f"❌ 系统错误：{error_msg}\n\n> 网络异常或服务暂时不可用，请稍后重试。"
        status = f"✗ 失败 · {type(e).__name__}"
        trace_summary = f"异常详情：{traceback.format_exc()[-500:]}"
    return result, status, trace_summary


def _format_trace(trace: dict) -> str:
    """格式化 Agent 推理跟踪为可读文本。"""
    if not trace:
        return ""
    lines = [f"**Trace ID:** {trace.get('trace_id', 'N/A')[:12]}..."]
    if trace.get("steps"):
        for s in trace["steps"]:
            thought = s.get("thought", "")[:80]
            action = s.get("action", "")
            latency = s.get("latency_ms", 0)
            success = "✓" if s.get("success") else "✗"
            lines.append(f"- {success} **{action}** ({latency:.0f}ms)")
            if thought:
                lines.append(f"  > {thought}")
    return "\n".join(lines)


def _safe_upload(file) -> str:
    """安全上传文件，返回格式化的结果消息。"""
    if file is None:
        return "⚠️ 请先选择文件"

    file_path = getattr(file, 'name', str(file))
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    format_icons = {
        ".pdf": "📄", ".docx": "📝", ".xlsx": "📊",
        ".pptx": "📽️", ".html": "🌐", ".htm": "🌐",
        ".epub": "📖", ".png": "🖼️", ".jpg": "🖼️",
        ".jpeg": "🖼️", ".gif": "🖼️", ".bmp": "🖼️",
        ".webp": "🖼️", ".tiff": "🖼️", ".tif": "🖼️",
        ".txt": "📃", ".md": "📃", ".py": "💻",
    }
    icon = format_icons.get(ext, "📎")

    try:
        start = time.perf_counter()
        result_json = rag_tool.execute("add_file", file_path=file_path)
        latency = (time.perf_counter() - start) * 1000

        import json
        parsed = json.loads(result_json)
        if parsed.get("ok"):
            data = parsed["data"]
            chunks = data.get("added_chunks", data.get("total_chunks", "?"))
            return f"{icon} **{file_name}** 导入成功 ✓\n\n" \
                   f"新增块数: {chunks}\n" \
                   f"文字总量: {rag_tool.pipeline.text_count}\n" \
                   f"图片总量: {rag_tool.pipeline.image_count}\n" \
                   f"耗时: {latency:.0f}ms"
        else:
            return f"{icon} **{file_name}** 导入失败 ✗\n\n{parsed.get('error', '未知错误')}"
    except json.JSONDecodeError:
        return f"{icon} **{file_name}** 处理完成\n\n{result_json[:300]}"
    except Exception as e:
        return f"❌ 网络或服务异常：{str(e)[:200]}\n\n请检查网络连接后重试。"



# gradio 6.x: css / theme 已从 Blocks 构造器移至 launch()(见文件底部)
with gr.Blocks(
    title="财务报销审核平台"
) as app:

    gr.HTML("""
    <div class="header">
        <div>
            <h1>🧾 财务报销审核平台</h1>
        </div>
        <div>
            <span class="status-dot"></span>
            <span class="status-text">系统运行中</span>
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes="tabs"):

        with gr.TabItem("💬 智能问答", id="chat"):
            def _session_choices():
                return [("＋ 新建会话", NEW_SESSION)] + [
                    (s["title"], s["id"]) for s in list_sessions()
                ]

            def _load_history(sid):
                if not sid or sid == NEW_SESSION:
                    return []
                return get_messages(sid)

            # 会话历史列表——新建 / 切换 / 删除
            with gr.Row(elem_classes="card"):
                session_dd = gr.Dropdown(
                    label="📋 最近会话",
                    choices=_session_choices(),
                    value=NEW_SESSION,
                    scale=4,
                )
                new_btn = gr.Button("＋ 新建", variant="secondary", scale=1)
                del_btn = gr.Button("🗑 删除", variant="stop", scale=1)

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=480,
                        avatar_images=(None, "🤖"),
                        elem_classes="chatbot",
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            label="",
                            placeholder="输入你的问题，例如：本季度营收与净利润的核心结论是什么？多来源数据是否一致？",
                            scale=4,
                            container=False,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Column(scale=1, visible=False) as trace_col:
                    trace_output = gr.Markdown("", elem_classes="card")

            with gr.Accordion("🔍 Agent 推理过程", open=False):
                trace_detail = gr.Markdown("")

            status_bar = gr.Markdown("")
            current_sid = gr.State(value=NEW_SESSION)

            def chat_with_trace(message, history, sid):
                history = history or []
                if not message or not message.strip():
                    return history, "", "", sid, gr.Dropdown(choices=_session_choices(), value=sid)
                if sid in (None, "", NEW_SESSION):
                    sid = create_session()
                answer, status, trace_summary = _safe_agent_run(message)
                save_exchange(sid, message, answer)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": answer})
                return history, status, trace_summary, sid, gr.Dropdown(choices=_session_choices(), value=sid)

            def on_new():
                sid = create_session()
                return [], "新会话已创建，开始提问吧", "", sid, gr.Dropdown(choices=_session_choices(), value=sid)

            def on_select(value):
                history = _load_history(value)
                sid = value if value and value != NEW_SESSION else NEW_SESSION
                title = value if value != NEW_SESSION else "新建会话"
                return history, f"已切换到会话：{title}", "", sid, gr.Dropdown(choices=_session_choices(), value=sid)

            def on_delete(sid):
                if not sid or sid == NEW_SESSION:
                    return [], "当前没有可删除的会话", "", NEW_SESSION, gr.Dropdown(choices=_session_choices(), value=NEW_SESSION)
                delete_session(sid)
                return [], "会话已删除", "", NEW_SESSION, gr.Dropdown(choices=_session_choices(), value=NEW_SESSION)

            outputs = [chatbot, status_bar, trace_detail, current_sid, session_dd]
            msg.submit(chat_with_trace, [msg, chatbot, current_sid], outputs)
            send_btn.click(chat_with_trace, [msg, chatbot, current_sid], outputs)
            new_btn.click(on_new, inputs=[], outputs=outputs)
            session_dd.select(on_select, [session_dd], outputs)
            del_btn.click(on_delete, [current_sid], outputs)

        with gr.TabItem("📤 文档管理", id="upload"):
            gr.Markdown("### 📤 文档管理\n\n上传文档入库，支持票据、发票、报销单、表格、图片等格式。")

            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="选择文件",
                        file_count="single",
                        file_types=[
                            ".pdf", ".docx", ".xlsx", ".pptx",
                            ".html", ".htm", ".epub",
                            ".txt", ".md", ".py", ".json", ".csv", ".xml",
                            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"
                        ],
                    )
                with gr.Column(scale=1):
                    upload_btn = gr.Button("导入知识库", variant="primary", size="lg")
                    gr.HTML('<div style="font-size:12px;color:#718096">'
                            '仅支持新版 Office 格式(.docx/.xlsx/.pptx)<br>'
                            '旧版(.doc/.xls/.ppt)请先另存为新格式</div>')

            upload_status = gr.Markdown("")

            upload_btn.click(_safe_upload, [file_input], [upload_status])

            gr.Markdown("---")
            gr.Markdown("### 知识库统计\n\n文档块与图片数量概览。")
            with gr.Row():
                text_stat = gr.Markdown("", elem_classes="stat-card")
                img_stat = gr.Markdown("", elem_classes="stat-card")
                total_stat = gr.Markdown("", elem_classes="stat-card")

            def refresh_stats():
                text_count = rag_tool.pipeline.text_count
                img_count = rag_tool.pipeline.image_count
                total = text_count + img_count
                return (
                    f"<div class='stat-card'><div class='stat-value'>{text_count:,}</div><div class='stat-label'>📝 文本块</div></div>",
                    f"<div class='stat-card'><div class='stat-value'>{img_count:,}</div><div class='stat-label'>🖼️ 图片</div></div>",
                    f"<div class='stat-card'><div class='stat-value'>{total:,}</div><div class='stat-label'>📦 总计</div></div>",
                )

            refresh_btn = gr.Button("刷新统计", variant="secondary", size="sm")
            refresh_btn.click(refresh_stats, outputs=[text_stat, img_stat, total_stat])

        with gr.TabItem("⚙️ 系统状态", id="system"):
            gr.Markdown("### ⚙️ 系统状态\n\n当前模型、存储与检索配置。")

            with gr.Row():
                with gr.Column():
                    sys_info = gr.Markdown(
                        f"- **向量模型**: paraphrase-multilingual-MiniLM (384维)\n"
                        f"- **视觉模型**: DeepSeek 多模态 (LLM_IMAGE_MODEL)\n"
                        f"- **文本模型**: DeepSeek 文本 (LLM_TEXT_MODEL)\n"
                        f"- **向量存储**: ChromaDB (HNSW 索引, 持久化)\n"
                        f"- **Agent 模式**: FC (优先) + ReAct (降级)\n"
                        f"- **检索策略**: MQE 多查询扩展 + CrossEncoder 精排"
                    )

                with gr.Column():
                    gr.Markdown(
                        "| 类别 | 格式 |\n"
                        "|------|------|\n"
                        "| 文档 | PDF, DOCX, EPUB |\n"
                        "| 表格 | XLSX, CSV |\n"
                        "| 演示 | PPTX |\n"
                        "| 网页 | HTML, HTM |\n"
                        "| 图片 | JPG, PNG, GIF, WebP, BMP, TIFF |\n"
                        "| 代码 | TXT, MD, PY, JSON, XML, LOG |"
                    )

    gr.HTML("""
    <footer>
        财务报销审核平台 · 规则优先 + LLM 兜底 · 多模态票据识别
    </footer>
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=BUSINESS_CSS,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    )
