"""A taskrunner that writes task definitions to a store"""

from time import time

from .base import Taskrunner
from .store_reader import StoreTaskReader


class StoreTaskWriter(Taskrunner):
    task_id: str

    def __init__(
        self,
        func,
        store,
        *args,
        read_store=None,
        pickle_func=False,
        get_result=None,
        **kwargs
    ):
        Taskrunner.__init__(self, func, *args, **kwargs)
        self._store = store
        self._get_result = get_result
        self._pickle_func = pickle_func
        if read_store:
            StoreTaskReader(store=read_store).listen()

    @property
    def result(self):
        if not self._task_id:
            return None
        if self._get_result:
            task_result = self._get_result(self._task_id)
        else:
            task_dict = self._store[self._task_id]
            task_result = task_dict.get('result', None)
        return task_result

    def __call__(self, *args, task_id: str = None, **kwargs):
        task_dict = {
            'name': self._func.__name__,
            'func': None,
            'init_args': list(self._init_args),
            'init_kwargs': dict(self._init_kwargs),
            'args': list(args),
            'kwargs': dict(kwargs),
            'start_time': int(time() * 1000),
            'task_status': 'new',
        }

        if self._pickle_func:
            # TODO: Make the serialization a configurable option, not hardcoded!
            try:
                import dill

                task_dict['func'] = dill.dumps(self._func)
            except ImportError:
                import pickle

                task_dict['func'] = pickle.dumps(self._func)

        if not task_id:
            import uuid

            task_id = str(uuid.uuid4())

        self._task_id = task_id
        self._store[task_id] = task_dict
