"""Tests for the Taskrunner base class and the coroutine taskrunner."""

import asyncio

import pytest

from strand.taskrunning.base import Taskrunner
from strand.taskrunning.coroutine import CoroutineTaskrunner


def test_base_runs_func_with_bound_and_call_args():
    assert Taskrunner(lambda a, b, c=0: a + b + c, 1, c=10)(2) == 13


def test_base_on_iter_and_on_end():
    seen, ended = [], []
    runner = Taskrunner(
        lambda n: iter(range(n)), on_iter=seen.append, on_end=ended.append
    )
    result = runner(3)
    assert seen == [0, 1, 2]
    assert ended == [result]


def test_base_error_handling():
    errors = []

    def boom():
        raise ValueError("x")

    assert Taskrunner(boom, on_error=errors.append)() is None
    assert isinstance(errors[0], ValueError)
    with pytest.raises(ValueError):
        Taskrunner(boom)()


def test_coroutine_call_and_callbacks():
    seen, ended = [], []
    runner = CoroutineTaskrunner(
        lambda n: iter(range(n)), on_iter=seen.append, on_end=ended.append
    )
    runner(3)
    assert seen == [0, 1, 2] and len(ended) == 1


def test_coroutine_get_coro_and_no_yield():
    runner = CoroutineTaskrunner(lambda a: a * 2, yield_on_iter=False)
    assert asyncio.run(runner.get_coro(4)) == 8
    seen = []
    runner = CoroutineTaskrunner(
        lambda: [1, 2], on_iter=seen.append, yield_on_iter=False
    )
    runner()
    assert seen == [1, 2]
