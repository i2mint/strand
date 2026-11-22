# strand.taskrunning.multi_dispatch

Defines the TaskDispatcher class.

### Classes

| [`TaskDispatcher`](#strand.taskrunning.multi_dispatch.TaskDispatcher)([target, check_complete])   | A class for dispatching multiple taskrunners and collecting their outputs.   |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|

### *class* strand.taskrunning.multi_dispatch.TaskDispatcher(target='thread', check_complete=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A class for dispatching multiple taskrunners and collecting their outputs.

Use the dispatch method to launch a task. Dispatch as many tasks as you want
with the same dispatcher, then check the property dispatcher.complete if you need
to confirm that all of the tasks have finished executing.

Results of tasks can be found in the ‘results’, ‘status’, and ‘errors’ properties.

```pycon
>>> from strand import TaskDispatcher
>>> from time import sleep
```

```pycon
>>> dispatcher = TaskDispatcher()
>>> def slow_double(x):
...     sleep(0.3)
...     return 2 * x
>>> _ = dispatcher.dispatch(slow_double, args=[21])
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
[42]
```

#### dispatch(func, \*init_args, target=None, on_iter=None, on_end=None, on_error=None, args=None, kwargs=None, \*\*init_kwargs)

Execute a single task. The results of the task will be appended to
the results, status, and errors lists.

#### run(timeout=10)

Execute all enqueued tasks.
