"""Defines the TaskDispatcher class."""

import time
from typing import Callable, Optional

from strand.taskrunning.base import Taskrunner
from strand.taskrunning.utils import resolve_runner_cls
from strand.constants import THREAD


class TaskDispatcher:
    """A class for dispatching multiple taskrunners and collecting their outputs.

    Use the dispatch method to launch a task. Dispatch as many tasks as you want
    with the same dispatcher, then check the property dispatcher.complete if you need
    to confirm that all of the tasks have finished executing.

    Results of tasks can be found in the 'results', 'status', and 'errors' properties.

    >>> from strand import TaskDispatcher
    >>> import requests
    >>> from time import sleep

    >>> dispatcher = TaskDispatcher()
    >>> _ = dispatcher.dispatch(requests.get, args=['https://zombo.com'])
    >>> dispatcher.status
    [None]
    >>> dispatcher.errors
    [None]
    >>> dispatcher.results
    [None]
    >>> sleep(1)
    >>> dispatcher.status
    ['success']
    >>> dispatcher.errors
    [None]
    >>> dispatcher.results
    [<Response [200]>]
    """

    _is_complete = False
    results = []
    status = []
    errors = []
    _queued_runners = []
    _dispatched_runners = []
    _target_cls = Taskrunner

    def __init__(self, target=THREAD, check_complete=None):
        """
        Initialize the dispatcher and set configuration.

        :param target: Optional - the default taskrunner target (a taskrunner class or a string
            ('thread', 'process', 'coroutine', or 'store'). Defaults to 'thread'.
        :param check_complete: Optional - a callable that evaluates whether all of the dispatcher's
            tasks are complete.
        """
        self._target_cls = resolve_runner_cls(target)
        self._check_complete = check_complete

    def enqueue(
        self,
        func: Callable,
        *init_args,
        target=None,
        on_iter: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        args=None,
        kwargs=None,
        **init_kwargs
    ):
        self._is_complete = False
        if not kwargs:
            kwargs = {}
        target_cls = resolve_runner_cls(target) if target else self._target_cls
        runner = target_cls(
            func,
            *init_args,
            on_iter=on_iter,
            on_end=self._mk_end_handler(on_end),
            on_error=self._mk_error_handler(on_error),
            **init_kwargs
        )
        self._queued_runners.append((runner, args, kwargs))

    def dispatch(
        self,
        func,
        *init_args,
        target=None,
        on_iter: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        args=None,
        kwargs=None,
        **init_kwargs
    ):
        """Execute a single task. The results of the task will be appended to
        the results, status, and errors lists.

        """
        self._is_complete = False
        if not kwargs:
            kwargs = {}
        target_cls = resolve_runner_cls(target) if target else self._target_cls
        runner = target_cls(
            func,
            *init_args,
            on_iter=on_iter,
            on_end=self._mk_end_handler(on_end),
            on_error=self._mk_error_handler(on_error),
            **init_kwargs
        )
        self._dispatched_runners.append(runner)
        return runner(*args, **kwargs)

    def run(self, timeout=10):
        """Execute all enqueued tasks.

        """
        while len(self._queued_runners):
            runner, args, kwargs = self._queued_runners.pop()
            self._dispatched_runners.append(runner)
            runner(*args, **kwargs)
        end_time = time.time() + timeout
        while time.time() < end_time:
            if self.complete:
                return True
            time.sleep(0.01)
        raise TimeoutError()

    def _mk_end_handler(self, on_end):
        target_index = len(self.results)
        self.results.append(None)
        self.status.append(None)

        def end_handler(result):
            self.results[target_index] = result
            self.status[target_index] = 'success'
            if on_end:
                on_end(result)

        return end_handler

    def _mk_error_handler(self, on_error):
        target_index = len(self.errors)
        self.errors.append(None)

        def error_handler(result):
            self.errors[target_index] = result
            self.status[target_index] = 'error'
            if on_error:
                on_error(result)

        return error_handler

    def reset(self):
        self._is_complete = False
        self.results = []
        self.status = []
        self.errors = []
        self._queued_runners = []
        self._dispatched_runners = []

    @property
    def complete(self):
        if self._is_complete:
            return True
        if self._check_complete:
            self._is_complete = self._check_complete(
                self.results, self.status, self.errors,
            )
        else:
            for status in self.status:
                if not status:
                    return False
            self._is_complete = True
        return self._is_complete
