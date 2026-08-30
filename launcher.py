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

if not os.path.exists(env_file):
    print("\n  ⚠ 首次启动，正在打开配置页面...")
    print("  填写后点击保存，服务将自动就绪\n")
    webbrowser.open(f"http://localhost:{API_PORT}/setup")
else:
    print("\n  ✅ 配置已就绪\n")
    # 拉起问答界面(Gradio,自足:直接走 Agent,不依赖 API)
    subprocess.Popen([sys.executable, "app.py"],
                     stdout=open("/dev/null", "w"), stderr=subprocess.STDOUT,
                     close_fds=True)
    # 拉起 API 服务(供外部集成 / curl 调用)
    subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app",
                      "--host", os.getenv("HOST", "0.0.0.0"), "--port", API_PORT],
                     stdout=open("/dev/null", "w"), stderr=subprocess.STDOUT,
                     close_fds=True)

    print(f"  🌐 问答界面  {UI_URL}")
    print(f"  🔌 API 服务  http://localhost:{API_PORT}")
    print("  ⚠  请勿关闭此窗口，关闭即停止服务；最小化即可")

    # 等 3 秒让 Gradio 起来，再自动打开问答界面
    time.sleep(3)
    webbrowser.open(UI_URL)

    try:
        input("按回车键停止所有服务并退出...\n")
    except (EOFError, KeyboardInterrupt):
        pass

print("\n服务已停止。")

