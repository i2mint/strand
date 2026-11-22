# strand.taskrunning.utils

Utilities

### Functions

| [`conditional_logger`](#strand.taskrunning.utils.conditional_logger)([verbose, log_func])          | Returns a logger function if verbose is True, otherwise a no-op function.   |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`resolve_runner_cls`](#strand.taskrunning.utils.resolve_runner_cls)(target)                       | Resolve a string or class to a Taskrunner subclass.                         |
| [`run_process`](#strand.taskrunning.utils.run_process)(func[, func_args, func_kwargs, ...]) | Context manager to launch a process and ensure cleanup.                     |

### strand.taskrunning.utils.conditional_logger(verbose=False, log_func=<built-in function print>)

Returns a logger function if verbose is True, otherwise a no-op function.
:type verbose: 
:param verbose: Whether to enable logging.
:type log_func: 
:param log_func: The logging function to use (default: print).

* **Returns:**
  A logging function.

### strand.taskrunning.utils.resolve_runner_cls(target)

Resolve a string or class to a Taskrunner subclass.
:type target: 
:param target: A string (e.g. ‘thread’, ‘process’, etc.) or a Taskrunner subclass.

* **Returns:**
  The resolved Taskrunner subclass.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If the target is not valid.

### strand.taskrunning.utils.run_process(func, func_args=(), func_kwargs=None, , process_name=None, is_ready=None, timeout=30, force_kill=True, verbose=False, process_already_running=None)

Context manager to launch a process and ensure cleanup.
If process_already_running is provided and returns True, the process is not launched or killed.
Otherwise, launches the process and kills it on exit (if force_kill).

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The function to run in a new process.
  * **func_args** – Positional arguments for the function.
  * **func_kwargs** – Keyword arguments for the function.
  * **process_name** – Name for the process (optional).
  * **is_ready** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int)) – Predicate or seconds to wait for readiness (optional).
  * **timeout** – Timeout for readiness check (seconds).
  * **force_kill** – Whether to kill the process on exit.
  * **verbose** – Enable verbose logging.
  * **process_already_running** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Callable that returns True if the process is already running.
* **Yields:**
  The process object if launched, else None.
