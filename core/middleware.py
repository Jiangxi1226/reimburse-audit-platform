import os, time
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from utils.security import RateLimiter



_limiter = RateLimiter(max_requests_per_minute=30)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件——单用户每分钟最多 N 次请求。

    限流键(client_id)优先取真实来源 IP。仅在明确信任反向代理时才采信
    X-Forwarded-For 头——而该头可被客户端伪造，直接采信首位 IP 会让攻击者
    改一个头即可绕过限流。因此这里仅在设置了 TRUST_PROXY 时启用 XFF 解析，
    否则一律用 TCP 连接的 peer 地址，保证限流不可被绕过。
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        # 默认用 socket 层对端地址(不可伪造);只有显式 TRUST_PROXY=1 才解析转发头
        client_id = request.client.host if request.client else "unknown"
        if os.getenv("TRUST_PROXY") == "1":
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                # 取首个 IP 需警惕伪造——但 TRUST_PROXY=1 意味着部署在前置代理后，
                # 代理会覆写该头，此时首位即真实客户端。
                client_id = forwarded.split(",")[0].strip()

        if not _limiter.allow(client_id):
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试。")

        return await call_next(request)



class APIKeyMiddleware(BaseHTTPMiddleware):
    """简易 API Key 认证中间件。

    生产环境应换 JWT + OAuth2，此处给面试做"权限控制"话题。
    """

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.enabled = bool(self.api_key)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        if request.url.path in self.SKIP_PATHS or request.url.path.startswith("/static"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        api_key = ""

        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        else:
            api_key = request.query_params.get("api_key", "")

        if api_key != self.api_key:
            raise HTTPException(status_code=401, detail="API Key 无效或缺失。请在 Header 中设置 Authorization: Bearer <your_key>。")

        return await call_next(request)



class RequestLogMiddleware(BaseHTTPMiddleware):
    """请求日志中间件——记录每个请求的路径、耗时、状态码。"""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        if response.status_code >= 400:
            try:
                from core.db import log_error
                log_error(
                    error_code=f"HTTP_{response.status_code}",
                    error_type="HTTPException",
                    message=f"{request.method} {request.url.path} → {response.status_code}",
                    context=f"耗时: {elapsed:.0f}ms",
                    severity="warning" if response.status_code < 500 else "error",
                )
            except Exception:
                pass

        return response
