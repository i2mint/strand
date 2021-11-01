"""Taskrunner base class"""

from typing import Callable, Optional


class Taskrunner:
    _on_iter: Optional[Callable]
    _on_end: Optional[Callable]
    _on_error: Optional[Callable]

    def __init__(
        self,
        func: Callable,
        *args,
        on_iter: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
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
