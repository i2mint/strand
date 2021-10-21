# strand
Easy creation of non-blocking tasks

To install:	```pip install strand```

## Warning

In order to use threads or multiprocessing safely, you need to understand the constraints of those features. A thorough discussion of how not to shoot yourself in the foot is outside the scope of this library. Future versions of this library may include strong input checks to prevent more common mistakes, with optional arguments to override checks if necessary. This version does not contain any safety controls yet.

## Basic Usage
```python
from strand import ThreadTaskrunner 

def handle_chunk(chunk):
    print(f'got a chunk: {chunk}')

def long_blocking_function(total_size, chunk_size):
    if total_size < chunk_size:
        total_size = chunk_size    
    big_list = range(total_size)
    return (big_list[chunk_size * n:chunk_size * (n + 1)] for n in range(total_size / chunk_size))

# instantiate the runner
runner = ThreadTaskrunner(long_blocking_function, on_iter=handle_chunk)

# call the runner with the arguments to pass to the function
# the function will run in a thread
runner(1e8, 1e3)
```

## Decorator syntax
```python
from strand import as_task 

def handle_chunk(chunk):
    print(f'got a chunk: {chunk}')

@as_task(on_iter=handle_chunk)
def long_blocking_function(total_size, chunk_size):
    if total_size < chunk_size:
        total_size = chunk_size    
    big_list = range(total_size)
    return (big_list[chunk_size * n:chunk_size * (n + 1)] for n in range(total_size / chunk_size)) 

# the function will run in a thread
long_blocking_function(1e8, 1e3)
```

The `as_task` decorator takes a taskrunner target as its first argument. The argument may be a Taskrunner subclass or a string. The allowed values are:
* `'thread'` (default): `ThreadTaskrunner`
* `'process'`: `MultiprocessTaskrunner`
* `'coroutine'`: `CoroutineTaskrunner`
* `'store'`: `StoreTaskWriter`
* `'sync'`: `Taskrunner` (just runs the function and returns the value synchronously without any change of context)

## Base API

`class strand.Taskrunner(func: Callable, *init_args, on_iter: Optional[Callable] = None,
on_end: Optional[Callable] = None, on_error: Optional[Callable] = None, **init_kwargs)`

The base Taskrunner class and its subclasses take a callable as their first init argument. Taskrunners implement `__call__` and pass arguments to their stored callable when called.

The `init_args` and `init_kwargs` are also passed to `func` when called (as `func(*init_args, *args, **init_kwargs, **kwargs)`, allowing a Taskrunner instance to serve as a partial invocation of a function.

The optional arguments `on_iter`, `on_end`, and `on_error` are callbacks to be invoked when applicable.
* If `on_iter` is provided and `func` returns an iterable, `on_iter` will be called with every item in the iterable after `func` returns.
* If `on_end` is provided, it will be called with the return value of `func`. Otherwise, for most subclasses, the return value of `func` will be discarded.
* If `on_error` is provided, it will be called with any exceptions thrown within `Taskrunner.__call__`. Otherwise, the taskrunner will re-throw exceptions after catching them.

## Subclasses

### ThreadTaskrunner
`class strand.ThreadTaskrunner(func: Callable, *init_args, on_iter: Optional[Callable],
on_end: Optional[Callable], on_error: Optional[Callable])`

Runs `func` in a thread. Simple as that.

### MultiprocessTaskrunner
`class strand.MultiprocessTaskrunner(func: Callable, *init_args, on_iter: Optional[Callable],
on_end: Optional[Callable], on_error: Optional[Callable], **init_kwargs)`

Runs `func` in a new process. Has a separate set of caveats from multi-threading.

### CoroutineTaskrunner
`class strand.MultiprocessTaskrunner(func: Callable, *init_args, on_iter: Optional[Callable],
on_end: Optional[Callable], on_error: Optional[Callable]), yield_on_iter: Optional[bool], **init_kwargs)`

Runs `func` in a coroutine. Requires the calling context to already be within a coroutine in order to derive much benefit. Not fully fleshed out yet.

If `yield_on_iter` is `True`, adds `await asyncio.sleep(0)` between every iteration, to yield control back to the coroutine scheduler.

## StoreTaskWriter
`class strand.StoreTaskWriter(func: Callable, store: Mapping, *init_args, on_iter: Optional[Callable], on_end: Optional[Callable], on_error: Optional[Callable]), read_store=None, pickle_func=False, get_result=None, **init_kwargs)`

When called, serializes `func` along with its arguments and passes them to `store` for storage, where it may then be found by a StoreTaskReader or any other consumer in another place and time.

The argument `read_store` takes a store that should expect to find values written in `store` and immediately instantiates a StoreTaskReader instance that starts polling `read_store` for items in a new thread.

If `pickle_func` is true, `func` will be serialized with `dill` for storage. Otherwise, only `func.__name__` will be stored (which should be enough for most use cases where the store reader knows as much as it should about the writer).

## StoreTaskReader (Not yet implemented)
`class strand.StoreTaskReader(store: Mapping, get_task_func: Optional[Callable])`

Accepts an argument `store` that should be a store of tasks to run.

The argument `get_task_func` should be a callable that resolves an item from the store into a function to call. If `get_task_func` is not present, the reader will assume that `store[some_key]['func']` is a pickled callable and will automatically attempt to unpickle it with `dill` before calling it with `*store[some_key]['args'], **store[some_key]['kwargs']`

Calling the `listen` method on a StoreTaskReader instance will cause it to start an infinite loop in a new thread to poll the store for new tasks and execute them. 
```python
reader = StoreTaskReader(task_store)

reader.listen()
```


## Future

* Taskrunners that dispatch tasks to network targets (e.g. MQTT, RabbitMQ, Redis)
  * Could just be a special case of store reader/writer
* Utilities for dispatching multiple tasks at once
* More customizable serialization
* Customize context for autogenerated StoreTaskReader when StoreTaskWriter is initialized with `read_store`
* Thorough/correct handling of coroutines (could be a whole library unto itself)
* Safety checking
