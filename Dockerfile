# 多模态 RAG 知识库助手 — Dockerfile
# 多阶段构建：构建阶段 + 运行阶段，减少镜像体积

# ═══ 构建阶段 ═══
FROM python:3.13-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ═══ 运行阶段 ═══
FROM python:3.13-slim

WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY . .

# 环境变量：Python 包路径
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 模型文件目录（运行时挂载宿主机模型到容器内）
# docker run -v /path/to/local_models:/app/../local_models ...
VOLUME ["/app/../local_models", "/app/chroma_data", "/app/data", "/app/logs", "/app/traces"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# 默认启动 Gradio Web UI（app.py）
# 如需启动 API 服务，运行: docker run ... multimodal-rag python api.py
CMD ["python", "app.py"]
