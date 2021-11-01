"""A taskrunner that calls functions in threads"""

from threading import Thread

from .base import Taskrunner


class ThreadTaskrunner(Taskrunner):
    _t = None

    def run(self, *args, **kwargs):
        Taskrunner.__call__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._t = Thread(target=self.run, args=args, kwargs=kwargs)
        self._t.setDaemon(True)
        on_end = self._on_end

        def handle_end_thread(result):
            if on_end:
                on_end(result)

        self._on_end = handle_end_thread
        self._t.start()
        return self._t
