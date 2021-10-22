"""A taskrunner that reads task definitions from a store and runs the tasks"""

from threading import Thread
from typing import Callable


class StoreTaskReader:
    _get_task_func: Callable

    def __init__(self, store, get_task_func=None):
        self._store = store
        self._get_task_func = get_task_func

    def _listen_handler(self):
        while True:
            available_tasks = list(self._store)
            # if available_tasks:
            #     for task in available_tasks:
            #         task_id = task['_id']
            #         if task['name'] in [
            #             'initiate_default_outlier_model',
            #             'continue_default_outlier_model',
            #         ]:
            #             continue
            #         func = available_task_types.get(task['name'], None)
            #         if not func:
            #             warn(
            #                 f'Task definition not found for task name {task["name"]}. Task cannot be run.'
            #             )
            #             update = {
            #                 'task_status': 'error',
            #                 'error': 'Task definition not found',
            #             }
            #         else:
            #             print(f'Running task: {str(task)}')
            #             task_manager.update_task_by_id(
            #                 task_id, {'task_status': 'running'}
            #             )
            #             try:
            #                 args = task['args']
            #                 kwargs = task['kwargs']
            #                 result = func(*args, **kwargs)
            #                 print(f'Task success: {not result.get("error", False)}')
            #                 update = {'task_status': 'complete'}
            #             except Exception:
            #                 error_trace = traceback.format_exc()
            #                 traceback.print_exc()
            #                 update = {
            #                     'task_status': 'error',
            #                     'error_trace': error_trace,
            #                 }
            #         task_manager.update_task_by_id(task_id, update)

    def listen(self):
        t = Thread(target=self._listen_handler)
        t.start()
