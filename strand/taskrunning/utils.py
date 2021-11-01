"""Utilities"""

from strand.taskrunning.base import Taskrunner
from strand.taskrunning.coroutine import CoroutineTaskrunner
from strand.taskrunning.multiprocess import MultiprocessTaskrunner
from strand.taskrunning.store_writer import StoreTaskWriter
from strand.taskrunning.thread import ThreadTaskrunner

from strand.constants import THREAD, PROCESS, SYNC, STORE, COROUTINE


def resolve_runner_cls(target):
    if isinstance(target, str):
        if target == THREAD:
            target_cls = ThreadTaskrunner
        elif target == PROCESS:
            target_cls = MultiprocessTaskrunner
        elif target == SYNC:
            target_cls = Taskrunner
        elif target == STORE:
            target_cls = StoreTaskWriter
        elif target == COROUTINE:
            target_cls = CoroutineTaskrunner
        else:
            raise ValueError(
                f'Taskrunner target {target} is invalid. '
                f'Valid targets are {THREAD}, {PROCESS}, {SYNC}, {STORE}, and {COROUTINE}.'
            )
    elif target == Taskrunner or issubclass(target, Taskrunner):
        target_cls = target
    else:
        raise ValueError(
            f'Taskrunner target {target} is invalid. Must be an allowed string or a Taskrunner class.'
        )
    return target_cls
