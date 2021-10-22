"""A taskrunner that calls functions in separate processes"""

from multiprocessing import Process, Queue

from .base import Taskrunner


class MultiprocessTaskrunner(Taskrunner):
    def run(self, *args, **kwargs):
        Taskrunner.__call__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        p = Process(target=self.run, args=args, kwargs=kwargs)
        p.start()
        return p
