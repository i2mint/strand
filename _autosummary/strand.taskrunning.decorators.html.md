# strand.taskrunning.decorators

Decorators to simplify taskrunner syntax

### Functions

| [`as_task`](#strand.taskrunning.decorators.as_task)([target])   | Creates a decorator to wrap a function in a taskrunner instance.   |
|----------------------------------------------------------------------|--------------------------------------------------------------------|

### strand.taskrunning.decorators.as_task(target='thread', \*\*kwargs)

Creates a decorator to wrap a function in a taskrunner instance.
When the decorator is called, the taskrunner will be launched
and the decorated function will be executed within the taskrunner’s target context.

* **Parameters:**
  * **target** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`type`](https://docs.python.org/3/builtins/functions.html#type)[`Taskrunner`]) – The kind of taskrunner to create. Valid targets are ‘thread’, ‘process’, ‘sync’,
    ‘store’, and ‘coroutine’.
  * **kwargs** – \_\_init_\_ arguments for the taskrunner.
* **Returns:**
  A decorator.

The most common default case is dispatching a function to a new thread, with no callbacks.

```pycon
>>> @as_task()
... def my_thread_function():
...     pass
```

You can add callbacks with keyword arguments.

```pycon
>>> def iter_callback(item):
...     pass
...
>>> @as_task(on_iter=iter_callback)
... def my_thread_iterable():
...     return []
...
>>> def end_callback(results):
...     pass
...
>>> @as_task(on_end=end_callback)
... def my_thread_iterable_2():
...     return []
```

You can specify a different kind of taskrunner. Please note that different taskrunners
have different constraints on the kinds of values they can handle (described in those classes).

```pycon
>>> @as_task('process', on_end=end_callback)
... def my_multiprocess_function():
...     pass
...
>>> @as_task('coroutine', yield_on_iter=False)
... def my_coroutine_function():
...     pass
...
```

This example will pickle the function and its arguments when invoked and save them in a store,
with the assumption that some other process will read from the same store and execute the function later.

```pycon
>>> @as_task('store', store=None, pickle_func=True)
... def my_stored_function():
...     pass
```
