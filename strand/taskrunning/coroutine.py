import asyncio

from .base import Taskrunner

class CoroutineTaskrunner(Taskrunner):
    _yield_on_iter: bool
    def __init__(self, *args, yield_on_iter=True, **kwargs):
        Taskrunner.__init__(self, *args, **kwargs)
        if yield_on_iter and self._on_iter:
            _on_iter = self._on_iter
            def async_on_iter(value):
                _on_iter(value)
                asyncio.sleep(0)
            self._on_iter = async_on_iter

    def __call__(self, *args, **kwargs):
        asyncio.run(Taskrunner.__call__(self, *args, **kwargs))
