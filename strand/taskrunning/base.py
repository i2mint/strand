"""Taskrunner base class"""

from typing import Optional
from collections.abc import Callable


class Taskrunner:
    _on_iter: Callable | None
    _on_end: Callable | None
    _on_error: Callable | None

    def __init__(
        self,
        func: Callable,
        *args,
        on_iter: Callable | None = None,
        on_end: Callable | None = None,
        on_error: Callable | None = None,
        **kwargs
    ):
        self._func = func
        self._on_iter = on_iter
        self._on_end = on_end
        self._on_error = on_error
        self._init_args = args
        self._init_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        try:
            result = self._func(*self._init_args, *args, **self._init_kwargs, **kwargs)
            if self._on_iter:
                for value in result:
                    self._on_iter(value)
            if self._on_end:
                self._on_end(result)
            return result
        except Exception as err:
            if self._on_error:
                self._on_error(err)
            else:
                raise err
