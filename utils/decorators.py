import time, functools






def timing(func):
    """测量函数执行时间。面试手搓概率 90%。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[{func.__name__}] {elapsed:.1f}ms")
        return result
    return wrapper




def retry(max_attempts: int = 3, delay: float = 0.5):
    """带参数装饰器——三层嵌套。面试手搓概率 70%。

    @retry(max_attempts=3, delay=0.5)
    def api_call():
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time as _time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    _time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator



def memoize(max_size: int = 128):
    """LRU 缓存装饰器——相同输入不重复计算。面试手搓概率 50%。

    用于：Embedding 编码——相同文本不重复调模型。
    """
    def decorator(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            if len(cache) >= max_size:
                cache.pop(next(iter(cache)))
            cache[key] = result
            return result
        return wrapper
    return decorator



def tool_action(func):
    """标记方法为可被 Tool.execute() 反射调用的操作。

    替代 base.py 中基于命名约定（_xxx）的动作发现——
    用装饰器显式标记，而非靠下划线前缀推断。

    面试对比：
      命名约定（当前项目）：隐式，简单但可能误匹配
      装饰器标记（此处）：显式，安全但需在每个方法上加
    """
    func._is_tool_action = True
    return func



def validate_args(**validators):
    """参数验证装饰器——在函数执行前检查参数合法性。

    @validate_args(query=lambda q: len(q) > 0, top_k=lambda k: 1 <= k <= 20)
    def search(query: str, top_k: int = 5):
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"{func.__name__}: 参数 {param_name}={value} 验证失败")
            return func(*args, **kwargs)
        return wrapper
    return decorator



"""
面试速记（三层嵌套的记忆方法）：
  最外层接收装饰器参数（n=3）
  中间层接收被装饰函数（func）
  最内层是替换函数（wrapper）

  @retry(n=3)
  def f(): ...

  = retry(n=3)(f)
  = decorator(f)
  = wrapper
"""
