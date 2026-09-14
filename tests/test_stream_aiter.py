"""异步流式适配器 _aiter_sync 测试。

背景：LLM 的 chat_stream 是同步阻塞生成器，直接在 async 端点里迭代会占住
事件循环、拖住并发请求。_aiter_sync 用「后台线程 + 队列」把它转成异步迭代。
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import _aiter_sync


def _slow_gen(n=5, delay=0.02):
    for i in range(n):
        time.sleep(delay)      # 模拟阻塞式拉取
        yield i


def test_order_preserved():
    async def run():
        return [x async for x in _aiter_sync(_slow_gen())]
    assert asyncio.run(run()) == [0, 1, 2, 3, 4]


def test_does_not_block_event_loop():
    """迭代期间事件循环应能做别的事（并发任务持续推进）。"""
    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        t = asyncio.create_task(ticker())
        out = [x async for x in _aiter_sync(_slow_gen(5, 0.03))]
        t.cancel()
        return ticks, out

    ticks, out = asyncio.run(run())
    assert out == [0, 1, 2, 3, 4]
    # 总耗时约 0.15s，ticker 每 5ms 一跳，若事件循环被阻塞则 ticks 会接近 0
    assert ticks >= 5, f"事件循环疑似被阻塞，ticks={ticks}"


def test_exception_propagates():
    def boom():
        yield 1
        raise RuntimeError("llm failed")

    async def run():
        got = []
        try:
            async for x in _aiter_sync(boom()):
                got.append(x)
        except RuntimeError as e:
            return got, str(e)
        return got, None

    got, err = asyncio.run(run())
    assert got == [1]
    assert err == "llm failed"


def test_should_stop_stops_pump():
    """should_stop 返回 True 时，后台线程应停止继续拉取。

    生成器刻意做成"慢速"（每项有停顿），贴近真实 LLM 流受网络限速的情形——
    若生成器零延迟，后台线程会在消费方来得及反应前就把它拉完，测不出止损。
    """
    pulled = []

    def slow_gen(n=100, delay=0.01):
        for i in range(n):
            time.sleep(delay)
            pulled.append(i)
            yield i

    flag = {"stop": False}

    async def run():
        out = []
        async for x in _aiter_sync(slow_gen(), should_stop=lambda: flag["stop"]):
            out.append(x)
            if len(out) >= 3:
                flag["stop"] = True
                break
        return out

    out = asyncio.run(run())
    assert out == [0, 1, 2]
    time.sleep(0.1)   # 给后台线程一点时间退出
    assert len(pulled) < 100, f"should_stop 未生效，线程拉取了 {len(pulled)} 项"
