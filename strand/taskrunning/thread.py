"""A taskrunner that calls functions in threads"""

from threading import Thread

from .base import Taskrunner

class ThreadTaskrunner(Taskrunner):
    def run(self, *args, **kwargs):
        Taskrunner.__call__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        t = Thread(target=self.run, args=args, kwargs=kwargs)
        if self._on_ended:
            _on_ended = self._on_ended
            def handle_end_thread(result):
                t.join()
                _on_ended(result)
            self._on_ended = handle_end_thread
        t.start()
        return t
