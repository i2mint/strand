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


# au.py - Asynchronous Computation Framework

A Python framework for transforming synchronous functions into asynchronous ones with status tracking, result persistence, and pluggable backends.

## Features

- 🚀 **Simple decorator-based API** - Transform any function into an async computation
- 💾 **Pluggable storage backends** - File system, Redis, databases, etc.
- 🔄 **Multiple execution backends** - Processes, threads, remote APIs
- 🛡️ **Middleware system** - Logging, metrics, authentication, rate limiting
- 🧹 **Automatic cleanup** - TTL-based expiration of old results
- 📦 **Flexible serialization** - JSON, Pickle, or custom formats
- 🔍 **Status tracking** - Monitor computation state and progress
- ❌ **Cancellation support** - Stop long-running computations


## Quick Start

```python
from strand.au import async_compute

@async_compute()
def expensive_computation(n: int) -> int:
    """Calculate factorial."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Launch computation (returns immediately)
handle = expensive_computation(100)

# Check status
print(handle.get_status())  # ComputationStatus.RUNNING

# Get result (blocks with timeout)
result = handle.get_result(timeout=30)
print(f"100! = {result}")
```

## Use Cases

### 1. **Long-Running Computations**
Perfect for computations that take minutes or hours:
- Machine learning model training
- Data processing pipelines
- Scientific simulations
- Report generation

### 2. **Web Application Background Tasks**
Offload heavy work from request handlers:
```python
@app.route('/analyze')
def analyze_data():
    handle = analyze_large_dataset(request.files['data'])
    return {'job_id': handle.key}

@app.route('/status/<job_id>')
def check_status(job_id):
    handle = ComputationHandle(job_id, store)
    return {'status': handle.get_status().value}
```

### 3. **Distributed Computing**
Use remote backends to distribute work:
```python
@async_compute(backend=RemoteAPIBackend(api_url="https://compute.example.com"))
def distributed_task(data):
    return complex_analysis(data)
```

### 4. **Batch Processing**
Process multiple items with shared infrastructure:
```python
store = FileSystemStore("/var/computations", ttl_seconds=3600)
backend = ProcessBackend(store)

@async_compute(backend=backend, store=store)
def process_item(item_id):
    return transform_item(item_id)

# Launch multiple computations
handles = [process_item(i) for i in range(1000)]
```

## Usage Patterns

### Basic Usage

```python
from strand.au import async_compute

# Simple async function with default settings
@async_compute()
def my_function(x):
    return x * 2

handle = my_function(21)
result = handle.get_result(timeout=10)  # Returns 42
```

### Custom Configuration

```python
from strand.au import async_compute, FileSystemStore, ProcessBackend
from strand.au import LoggingMiddleware, MetricsMiddleware, SerializationFormat

# Configure store with TTL and serialization
store = FileSystemStore(
    "/var/computations",
    ttl_seconds=3600,  # 1 hour TTL
    serialization=SerializationFormat.PICKLE  # For complex objects
)

# Add middleware
middleware = [
    LoggingMiddleware(level=logging.INFO),
    MetricsMiddleware()
]

# Create backend with middleware
backend = ProcessBackend(store, middleware=middleware)

# Apply to function
@async_compute(backend=backend, store=store)
def complex_computation(data):
    return analyze(data)
```

### Shared Infrastructure

```python
# Create shared components
store = FileSystemStore("/var/shared", ttl_seconds=7200)
backend = ProcessBackend(store)

# Multiple functions share the same infrastructure
@async_compute(backend=backend, store=store)
def step1(x):
    return preprocess(x)

@async_compute(backend=backend, store=store)
def step2(x):
    return transform(x)

# Chain computations
data = load_data()
h1 = step1(data)
preprocessed = h1.get_result(timeout=60)
h2 = step2(preprocessed)
final_result = h2.get_result(timeout=60)
```

### Temporary Computations

```python
from strand.au import temporary_async_compute

# Automatic cleanup when context exits
with temporary_async_compute(ttl_seconds=60) as async_func:
    @async_func
    def quick_job(x):
        return x ** 2
    
    handle = quick_job(10)
    result = handle.get_result(timeout=5)
    # Temporary directory cleaned up automatically
```

### Thread Backend for I/O-Bound Tasks

```python
from strand.au import ThreadBackend

# Use threads for I/O-bound operations
store = FileSystemStore("/tmp/io_tasks")
backend = ThreadBackend(store)

@async_compute(backend=backend, store=store)
def fetch_data(url):
    return requests.get(url).json()

# Launch multiple I/O operations
handles = [fetch_data(url) for url in urls]
```

## Architecture & Design

### Core Components

1. **Storage Abstraction (`ComputationStore`)**
   - Implements Python's `MutableMapping` interface
   - Handles result persistence and retrieval
   - Supports TTL-based expiration
   - Extensible for any storage backend

2. **Execution Abstraction (`ComputationBackend`)**
   - Defines how computations are launched
   - Supports different execution models
   - Integrates middleware for cross-cutting concerns

3. **Result Handling (`ComputationHandle`)**
   - Clean API for checking status and retrieving results
   - Supports timeouts and cancellation
   - Provides access to metadata

4. **Middleware System**
   - Lifecycle hooks: before, after, error
   - Composable and reusable
   - Examples: logging, metrics, auth, rate limiting

### Design Principles

- **Separation of Concerns**: Storage, execution, and result handling are independent
- **Dependency Injection**: All components are injected, avoiding hardcoded dependencies
- **Open/Closed Principle**: Extend functionality without modifying core code
- **Standard Interfaces**: Uses Python's `collections.abc` interfaces
- **Functional Approach**: Decorator-based API preserves function signatures

### Trade-offs & Considerations

#### Pros
- ✅ Clean abstraction allows easy swapping of implementations
- ✅ Type hints and dataclasses provide excellent IDE support
- ✅ Follows SOLID principles for maintainability
- ✅ Minimal dependencies (uses only Python stdlib)
- ✅ Flexible serialization supports complex objects
- ✅ Middleware enables cross-cutting concerns

#### Cons
- ❌ Process-based backend has overhead for small computations
- ❌ File-based storage might not scale for high throughput
- ❌ Metrics middleware doesn't share state across processes by default
- ❌ No built-in distributed coordination
- ❌ Fork method required for ProcessBackend (platform-specific)

#### When to Use
- ✅ Long-running computations (minutes to hours)
- ✅ Need to persist results across restarts
- ✅ Want to separate computation from result retrieval
- ✅ Building async APIs or job queues
- ✅ Need cancellation or timeout support

#### When NOT to Use
- ❌ Sub-second computations (overhead too high)
- ❌ Need distributed coordination (use Celery/Dask)
- ❌ Require complex workflow orchestration
- ❌ Need real-time streaming results

## Advanced Features

### Custom Middleware

```python
from strand.au import Middleware

class RateLimitMiddleware(Middleware):
    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self.requests = []
    
    def before_compute(self, func, args, kwargs, key):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < 60]
        
        if len(self.requests) >= self.max_per_minute:
            raise Exception("Rate limit exceeded")
        
        self.requests.append(now)
    
    def after_compute(self, key, result):
        pass
    
    def on_error(self, key, error):
        pass

# Use the middleware
@async_compute(middleware=[RateLimitMiddleware(max_per_minute=10)])
def rate_limited_function(x):
    return expensive_api_call(x)
```

### Custom Storage Backend

```python
from strand.au import ComputationStore, ComputationResult
import redis

class RedisStore(ComputationStore):
    def __init__(self, redis_client, *, ttl_seconds=None):
        super().__init__(ttl_seconds=ttl_seconds)
        self.redis = redis_client
    
    def create_key(self):
        return f"computation:{uuid.uuid4()}"
    
    def __getitem__(self, key):
        data = self.redis.get(key)
        if data is None:
            return ComputationResult(None, ComputationStatus.PENDING)
        return pickle.loads(data)
    
    def __setitem__(self, key, result):
        data = pickle.dumps(result)
        if self.ttl_seconds:
            self.redis.setex(key, self.ttl_seconds, data)
        else:
            self.redis.set(key, data)
    
    def __delitem__(self, key):
        self.redis.delete(key)
    
    def __iter__(self):
        return iter(self.redis.scan_iter("computation:*"))
    
    def __len__(self):
        return len(list(self))
    
    def cleanup_expired(self):
        # Redis handles expiration automatically
        return 0

# Use Redis backend
redis_client = redis.Redis(host='localhost', port=6379)
store = RedisStore(redis_client, ttl_seconds=3600)

@async_compute(store=store)
def distributed_computation(x):
    return process(x)
```

### Monitoring & Metrics

```python
from strand.au import MetricsMiddleware

# Create shared metrics
metrics = MetricsMiddleware()

@async_compute(middleware=[metrics])
def monitored_function(x):
    return compute(x)

# Launch several computations
for i in range(10):
    monitored_function(i)

# Check metrics
stats = metrics.get_stats()
print(f"Total: {stats['total']}")
print(f"Completed: {stats['completed']}")
print(f"Failed: {stats['failed']}")
print(f"Avg Duration: {stats['avg_duration']:.2f}s")
```

## Error Handling

```python
@async_compute()
def may_fail(x):
    if x < 0:
        raise ValueError("x must be positive")
    return x ** 2

handle = may_fail(-5)

try:
    result = handle.get_result(timeout=5)
except Exception as e:
    print(f"Computation failed: {e}")
    print(f"Status: {handle.get_status()}")  # ComputationStatus.FAILED
```

## Cleanup Strategies

```python
# Manual cleanup
@async_compute(ttl_seconds=3600)
def my_func(x):
    return x * 2

# Clean up expired results
removed = my_func.cleanup_expired()
print(f"Removed {removed} expired results")

# Automatic cleanup with probability
store = FileSystemStore(
    "/tmp/computations",
    ttl_seconds=3600,
    auto_cleanup=True,
    cleanup_probability=0.1  # 10% chance on each access
)
```

## API Reference

### Main Decorator

```python
@async_compute(
    backend=None,           # Execution backend (default: ProcessBackend)
    store=None,            # Storage backend (default: FileSystemStore)
    base_path="/tmp/computations",  # Path for default file store
    ttl_seconds=3600,      # Time-to-live for results
    serialization=SerializationFormat.JSON,  # JSON or PICKLE
    middleware=None        # List of middleware components
)
```

### ComputationHandle Methods

- `is_ready() -> bool`: Check if computation is complete
- `get_status() -> ComputationStatus`: Get current status
- `get_result(timeout=None) -> T`: Get result, optionally wait
- `cancel() -> bool`: Attempt to cancel computation
- `metadata -> Dict[str, Any]`: Access computation metadata

### ComputationStatus Enum

- `PENDING`: Not started yet
- `RUNNING`: Currently executing
- `COMPLETED`: Successfully finished
- `FAILED`: Failed with error

## Process Management Utility

The `run_process` context manager allows you to launch a process for the duration of a context, with optional readiness checks and automatic cleanup. It is useful for ensuring a background service or worker is running for the duration of a test or script.

**Example:**

```python
from strand.taskrunning.utils import run_process
import time

def my_worker():
    print("Worker started!")
    time.sleep(5)
    print("Worker exiting!")

with run_process(my_worker, process_name="my_worker", is_ready=0.2, timeout=5) as proc:
    print(f"Process running: {proc.is_alive()}")
    # Do work while the process is running
    time.sleep(1)
# After the context, the process is cleaned up
```

You can also use the `process_already_running` argument to avoid launching a process if an external check indicates it is already running:

```python
def is_service_running():
    # Return True if the service is already running
    ...

with run_process(my_worker, process_already_running=is_service_running) as proc:
    if proc is None:
        print("Service was already running!")
    else:
        print("Launched new worker process!")
```
