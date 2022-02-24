"""Utilities"""

from contextlib import contextmanager
from multiprocessing import Process, active_children
from time import sleep, time
from typing import Any, Callable, Union
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


def conditional_logger(verbose=False, log_func=print):
    if verbose:
        return log_func
    else:

        def clog(*args, **kwargs):
            pass  # do nothing

        return clog


@contextmanager
def run_process(
    func: Callable,
    func_args=(),
    func_kwargs=None,
    process_name=None,
    is_ready: Union[Callable[[], Any], float, int] = None,
    timeout=30,
    force_kill=True,
    verbose=False,
):
    def launch_process():
        try:
            print('starting process!...')
            clog(f'Starting {process_name} process...')
            process.start()
            clog(f'... {process_name} process started.')
        except Exception:
            raise RuntimeError(
                f'Something went wrong when trying to launch process {process_name}'
            )

    def launch_and_wait_till_ready(
        start_process: Callable[[], Any],
        is_ready: Union[Callable[[], Any], float, int] = 5.0,
        check_every_seconds=1.0,
        timeout=30.0,
    ):
        """A function that launches a process, checks if it's ready, and exits when it is.

        :param start_process: A argument-less function that launches an independent process
        :param is_ready: A argument-less function that returns False if, and only if, the process should be considered ready
        :param check_every_seconds: Determines the frequency that is_ready will be called
        :param timeout: Determines how long to wait for the process to be ready before we should give up
        :return: start_process_output, is_ready_output
        """
        start_time = time()

        # If is_ready is a number, make an is_ready function out of it
        if isinstance(is_ready, (float, int)):
            is_ready_in_seconds = is_ready

            def is_ready_func():
                f"""Returns True if, and only if, {is_ready_in_seconds} elapsed"""
                return time() - start_time >= is_ready_in_seconds

            is_ready_func.__name__ = f'wait_for_seconds({is_ready_in_seconds})'
            is_ready = is_ready_func
        start_process_output = start_process()  # needs launch a parallel process!
        while time() - start_time < timeout:
            tic = time()
            is_ready_output = is_ready()
            if is_ready_output is False:
                elapsed = time() - tic
                sleep(max(0, check_every_seconds - elapsed))
            else:
                return start_process_output, is_ready_output
        # If you got so far, raise TimeoutError
        raise TimeoutError(
            f"Launching {getattr(start_process, '__qualname__', None)} "
            f"and checking for readiness with {getattr(is_ready, '__qualname__', None)} "
            f'timedout (timeout={timeout}s)'
        )

    kwargs = func_kwargs or {}
    clog = conditional_logger(verbose)
    process_name = process_name or getattr(func, '__qualname__', '\b')

    try:
        process = Process(target=func, args=func_args, kwargs=kwargs, name=process_name)

        if is_ready:  # if the 'is_ready' time or predicate is defined
            launch_and_wait_till_ready(launch_process, is_ready, timeout=timeout)
        else:
            launch_process()

        yield process
    finally:
        if process is not None and process.is_alive():
            if force_kill:
                clog(f'Killing process: {process_name}...')
                for child in active_children():
                    clog(child)
                    child.kill()
                # clog(process)
                # process.kill()
                clog(f'... {process_name} process killed')
            else:
                process.join()
