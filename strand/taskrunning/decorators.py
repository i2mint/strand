"""Decorators to simplify taskrunner syntax"""

from functools import wraps
from typing import Union, Type

from strand.taskrunning.base import Taskrunner
from strand.constants import THREAD
from strand.taskrunning.utils import resolve_runner_cls


def as_task(target: Union[str, Type[Taskrunner]] = THREAD, **kwargs):
    """Creates a decorator to wrap a function in a taskrunner instance.
    When the decorator is called, the taskrunner will be launched
    and the decorated function will be executed within the taskrunner's target context.

    :param target: The kind of taskrunner to create. Valid targets are 'thread', 'process', 'sync',
        'store', and 'coroutine'.
    :param kwargs: __init__ arguments for the taskrunner.
    :return: A decorator.

    The most common default case is dispatching a function to a new thread, with no callbacks.

    >>> @as_task()
    ... def my_thread_function():
    ...     pass

    You can add callbacks with keyword arguments.

    >>> def iter_callback(item):
    ...     pass
    ...
    >>> @as_task(on_iter=iter_callback)
    ... def my_thread_iterable():
    ...     return []
    ...
    >>> def end_callback(results):
    ...     pass
    ...
    >>> @as_task(on_end=end_callback)
    ... def my_thread_iterable_2():
    ...     return []

    You can specify a different kind of taskrunner. Please note that different taskrunners
    have different constraints on the kinds of values they can handle (described in those classes).
    >>> @as_task('process', on_end=end_callback)
    ... def my_multiprocess_function():
    ...     pass
    ...
    >>> @as_task('coroutine', yield_on_iter=False)
    ... def my_coroutine_function():
    ...     pass
    ...

    This example will pickle the function and its arguments when invoked and save them in a store,
    with the assumption that some other process will read from the same store and execute the function later.
    >>> @as_task('store', store=None, pickle_func=True)
    ... def my_stored_function():
    ...     pass
    """

    def deco(func):
        target_cls = resolve_runner_cls(target)
        return wraps(func)(target_cls(func, **kwargs))

    return deco
