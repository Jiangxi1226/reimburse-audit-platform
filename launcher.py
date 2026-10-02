import sys, os, webbrowser, subprocess, time


def _load_env(env_file: str) -> None:
    """把 .env 里的 KEY=VALUE 注入当前进程环境（setdefault，不覆盖已有的）。

    项目约定 api.py 的 .env 只当"配置存档"、不会自动注入进程，
    但端口/主机位必须由启动器读出来透传给 uvicorn——否则 PORT 改了不生效。
    subprocess 会继承本进程 os.environ，因此这里注入后子进程 uvicorn 也能读到。
    """
    if not os.path.exists(env_file):
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


os.chdir(os.path.dirname(os.path.abspath(__file__)))
env_file = os.path.join(os.path.dirname(__file__), ".env")
_load_env(env_file)

# API 服务端口(由 .env 决定,默认 8000);问答界面(Gradio)固定 7860
API_PORT = os.getenv("PORT", "8000")
UI_PORT = os.getenv("GRADIO_PORT", "7860")
UI_URL = f"http://localhost:{UI_PORT}"

print("=" * 52)
print("  财务报销审核平台")
print("=" * 52)

# 判断依据是"有没有可用的 LLM 配置"，而非".env 文件是否存在"：
# 文件在但 LLM_API_KEY 为空时，Gradio/Agent 初始化会失败，只能走 /setup 补配。
_needs_setup = not os.getenv("LLM_API_KEY")

# 无论是否已配置都先拉起 API：未配置时它以"降级模式"运行，仅 /setup 等配置类端点可用。
subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app",
                  "--host", os.getenv("HOST", "0.0.0.0"), "--port", API_PORT],
                 stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT,
                 close_fds=True)

if _needs_setup:
    print("\n  ⚠ 尚未配置 LLM，正在打开配置页面...")
    print("  填写后点击保存，服务将自动就绪，无需重启\n")
    time.sleep(2)
    webbrowser.open(f"http://localhost:{API_PORT}/setup")
else:
    print("\n  ✅ 配置已就绪\n")
    # 拉起问答界面(Gradio,自足:直接走 Agent,不依赖 API)
    subprocess.Popen([sys.executable, "app.py"],
                     stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT,
                     close_fds=True)
    print(f"  🌐 问答界面  {UI_URL}")
    print(f"  🔌 API 服务  http://localhost:{API_PORT}")
    time.sleep(3)
    webbrowser.open(UI_URL)

print("  ⚠  请勿关闭此窗口，关闭即停止服务；最小化即可")

try:
    input("\n按回车键停止所有服务并退出...\n")
except (EOFError, KeyboardInterrupt):
    pass

print("\n服务已停止。")
