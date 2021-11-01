"""A taskrunner that calls functions in coroutines"""

import asyncio

from .base import Taskrunner


class CoroutineTaskrunner(Taskrunner):
    _yield_on_iter: bool

    def __init__(self, *args, yield_on_iter=True, **kwargs):
        Taskrunner.__init__(self, *args, **kwargs)
        self._yield_on_iter = yield_on_iter

    async def acall(self, *args, **kwargs):
        result = self._func(*args, **kwargs)
        if self._on_iter:
            for value in result:
                self._on_iter(value)
                if self._yield_on_iter:
                    await asyncio.sleep(0)
        if self._on_end:
            self._on_end(result)
        return result

    def get_coro(self, *args, **kwargs):
        return self.acall(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return asyncio.run(self.acall(*args, **kwargs))
