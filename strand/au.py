"""
Asynchronous computation framework with pluggable backends.

This module provides a decorator-based approach to transform synchronous functions
into asynchronous ones that return immediately with a handle to check status
and retrieve results later.

Architecture Overview:
--------------------
1. Storage abstraction (ComputationStore): Implements MutableMapping interface
2. Execution abstraction (ComputationBackend): Handles computation launching
3. Result handling (ComputationHandle): Clean API for status/result retrieval
4. Middleware system: For cross-cutting concerns like logging and metrics
5. Cleanup mechanisms: Automatic expiration of old results

Usage Patterns:
--------------
Simple file-based async:
    >>> @async_compute()
    ... def my_function(x):
    ...     return x * 2

Custom configuration (example setup):
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     store = FileSystemStore(tmpdir, ttl_seconds=3600)
    ...     backend = ProcessBackend(store, middleware=[LoggingMiddleware(), MetricsMiddleware()])
    ...     @async_compute(backend=backend, store=store)
    ...     def api_computation(data):
    ...         return data * 2
    ...     # Function is ready to use
    ...     True
    True

Shared infrastructure:
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     store = FileSystemStore(tmpdir, ttl_seconds=3600)
    ...     backend = ProcessBackend(store)
    ...     @async_compute(backend=backend, store=store)
    ...     def step1(x):
    ...         return x + 1
    ...     @async_compute(backend=backend, store=store)
    ...     def step2(x):
    ...         return x * 2
    ...     # Functions are ready to use
    ...     True
    True
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union, List, Dict
import json
import logging
import multiprocessing
import pickle
import time
import uuid
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class ComputationStatus(Enum):
    """Status of a computation task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SerializationFormat(Enum):
    """Supported serialization formats."""

    JSON = "json"
    PICKLE = "pickle"


@dataclass
class ComputationResult:
    """
    Result of a computation with metadata.

    >>> result = ComputationResult("done", ComputationStatus.COMPLETED)
    >>> result.is_ready
    True
    """

    value: Any
    status: ComputationStatus
    error: Optional[Exception] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Check if computation is complete (success or failure)."""
        return self.status in (ComputationStatus.COMPLETED, ComputationStatus.FAILED)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get computation duration if completed."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return None


T = TypeVar("T")


class ComputationStore(MutableMapping[str, ComputationResult], ABC):
    """
    Abstract base for storing computation results with TTL support.

    Stores should implement cleanup of expired results.
    """

    def __init__(self, *, ttl_seconds: Optional[int] = None):
        """
        Initialize store with optional TTL.

        Args:
            ttl_seconds: Time-to-live for results in seconds
        """
        self.ttl_seconds = ttl_seconds

    @abstractmethod
    def create_key(self) -> str:
        """Generate a unique key for a new computation."""
        pass

    def is_expired(self, result: ComputationResult) -> bool:
        """Check if a result has expired based on TTL."""
        if self.ttl_seconds is None or not result.is_ready:
            return False

        expiry_time = result.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired results. Returns count of removed items."""
        pass


class FileSystemStore(ComputationStore):
    """
    Store computation results in the filesystem with automatic cleanup.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     store = FileSystemStore(tmpdir, ttl_seconds=3600)
    ...     key = store.create_key()
    ...     store[key] = ComputationResult(42, ComputationStatus.COMPLETED)
    ...     store[key].value
    42
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        *,
        suffix: str = ".json",
        ttl_seconds: Optional[int] = None,
        serialization: SerializationFormat = SerializationFormat.JSON,
        auto_cleanup: bool = True,
        cleanup_probability: float = 0.1,
    ):
        super().__init__(ttl_seconds=ttl_seconds)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self.serialization = serialization
        self.auto_cleanup = auto_cleanup
        self.cleanup_probability = cleanup_probability

    def create_key(self) -> str:
        """Generate a unique filename."""
        return str(uuid.uuid4())

    def _get_path(self, key: str) -> Path:
        """Get the full path for a key."""
        return self.base_path / f"{key}{self.suffix}"

    def _maybe_cleanup(self) -> None:
        """Randomly trigger cleanup based on probability."""
        if self.auto_cleanup and self.ttl_seconds:
            import random

            if random.random() < self.cleanup_probability:
                self.cleanup_expired()

    def _serialize(self, result: ComputationResult) -> bytes:
        """Serialize result based on configured format."""
        data = {
            "value": result.value,
            "status": result.status.value,
            "error": (
                {
                    "type": type(result.error).__name__,
                    "message": str(result.error),
                    "module": type(result.error).__module__,
                }
                if result.error
                else None
            ),
            "created_at": result.created_at.isoformat(),
            "completed_at": (
                result.completed_at.isoformat() if result.completed_at else None
            ),
            "metadata": result.metadata,
        }

        if self.serialization == SerializationFormat.JSON:
            return json.dumps(data).encode()
        else:  # PICKLE
            return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> ComputationResult:
        """Deserialize result based on configured format."""
        if self.serialization == SerializationFormat.JSON:
            obj = json.loads(data.decode())
        else:  # PICKLE
            obj = pickle.loads(data)

        # Reconstruct error with proper type
        error = None
        if obj.get("error"):
            error_info = obj["error"]
            if isinstance(error_info, str):
                # Legacy format - just use Exception
                error = Exception(error_info)
            else:
                # New format with type information
                error_type = error_info.get("type", "Exception")
                error_message = error_info.get("message", "")

                # Try to get the actual exception class
                try:
                    if error_type == "ValueError":
                        error = ValueError(error_message)
                    elif error_type == "RuntimeError":
                        error = RuntimeError(error_message)
                    elif error_type == "TimeoutError":
                        error = TimeoutError(error_message)
                    else:
                        error = Exception(error_message)
                except Exception:
                    # Fallback to generic Exception
                    error = Exception(error_message)

        return ComputationResult(
            value=obj.get("value"),
            status=ComputationStatus(obj["status"]),
            error=error,
            created_at=datetime.fromisoformat(obj["created_at"]),
            completed_at=(
                datetime.fromisoformat(obj["completed_at"])
                if obj.get("completed_at")
                else None
            ),
            metadata=obj.get("metadata", {}),
        )

    def __getitem__(self, key: str) -> ComputationResult:
        self._maybe_cleanup()
        path = self._get_path(key)

        if not path.exists():
            return ComputationResult(None, ComputationStatus.PENDING)

        try:
            with open(path, "rb") as f:
                return self._deserialize(f.read())
        except Exception as e:
            logger.error(f"Failed to read result for key {key}: {e}")
            return ComputationResult(None, ComputationStatus.FAILED, error=e)

    def __setitem__(self, key: str, result: ComputationResult) -> None:
        path = self._get_path(key)

        # Update completion time if transitioning to ready state
        if result.is_ready and result.completed_at is None:
            result.completed_at = datetime.now()

        try:
            with open(path, "wb") as f:
                f.write(self._serialize(result))
        except Exception as e:
            logger.error(f"Failed to write result for key {key}: {e}")

    def __delitem__(self, key: str) -> None:
        self._get_path(key).unlink(missing_ok=True)

    def __iter__(self):
        return (p.stem for p in self.base_path.glob(f"*{self.suffix}"))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def cleanup_expired(self) -> int:
        """Remove expired results from filesystem."""
        removed = 0
        for key in list(self):  # Create list to avoid modification during iteration
            try:
                result = self[key]
                if self.is_expired(result):
                    del self[key]
                    removed += 1
                    logger.debug(f"Cleaned up expired result: {key}")
            except Exception as e:
                logger.error(f"Error during cleanup of {key}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired results")

        return removed


class Middleware(ABC):
    """Base class for middleware components."""

    @abstractmethod
    def before_compute(
        self, func: Callable, args: tuple, kwargs: dict, key: str
    ) -> None:
        """Called before computation starts."""
        pass

    @abstractmethod
    def after_compute(self, key: str, result: ComputationResult) -> None:
        """Called after computation completes."""
        pass

    @abstractmethod
    def on_error(self, key: str, error: Exception) -> None:
        """Called when computation fails."""
        pass


class LoggingMiddleware(Middleware):
    """
    Middleware for logging computation lifecycle.

    >>> middleware = LoggingMiddleware(level=logging.INFO)
    """

    def __init__(
        self, *, level: int = logging.DEBUG, logger_name: Optional[str] = None
    ):
        self.level = level
        self.logger = logging.getLogger(logger_name or __name__)

    def before_compute(
        self, func: Callable, args: tuple, kwargs: dict, key: str
    ) -> None:
        """Log computation start."""
        self.logger.log(self.level, f"Starting computation {key} for {func.__name__}")

    def after_compute(self, key: str, result: ComputationResult) -> None:
        """Log computation completion."""
        duration = result.duration
        duration_str = f" in {duration.total_seconds():.2f}s" if duration else ""
        self.logger.log(self.level, f"Completed computation {key}{duration_str}")

    def on_error(self, key: str, error: Exception) -> None:
        """Log computation error."""
        self.logger.error(f"Computation {key} failed: {error}", exc_info=True)


class MetricsMiddleware(Middleware):
    """
    Middleware for collecting computation metrics.

    >>> metrics = MetricsMiddleware()
    >>> metrics.get_stats()
    {'total': 0, 'completed': 0, 'failed': 0, 'avg_duration': 0.0}
    """

    def __init__(self):
        self.total_computations = 0
        self.completed_computations = 0
        self.failed_computations = 0
        self.total_duration = 0.0
        self._lock = multiprocessing.Lock()

    def before_compute(
        self, func: Callable, args: tuple, kwargs: dict, key: str
    ) -> None:
        """Increment total computations."""
        with self._lock:
            self.total_computations += 1

    def after_compute(self, key: str, result: ComputationResult) -> None:
        """Update completion metrics."""
        with self._lock:
            self.completed_computations += 1
            if result.duration:
                self.total_duration += result.duration.total_seconds()

    def on_error(self, key: str, error: Exception) -> None:
        """Update failure metrics."""
        with self._lock:
            self.failed_computations += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            avg_duration = (
                self.total_duration / self.completed_computations
                if self.completed_computations > 0
                else 0.0
            )
            return {
                "total": self.total_computations,
                "completed": self.completed_computations,
                "failed": self.failed_computations,
                "avg_duration": avg_duration,
            }


class SharedMetricsMiddleware(Middleware):
    """Metrics middleware using shared memory for multiprocessing."""

    def __init__(self):
        # Use multiprocessing.Value for shared counters
        self.total_computations = multiprocessing.Value("i", 0)
        self.completed_computations = multiprocessing.Value("i", 0)
        self.failed_computations = multiprocessing.Value("i", 0)
        self.total_duration = multiprocessing.Value("d", 0.0)
        self._lock = multiprocessing.Lock()

    def before_compute(
        self, func: Callable, args: tuple, kwargs: dict, key: str
    ) -> None:
        with self._lock:
            self.total_computations.value += 1

    def after_compute(self, key: str, result: ComputationResult) -> None:
        with self._lock:
            self.completed_computations.value += 1
            if result.duration:
                self.total_duration.value += result.duration.total_seconds()

    def on_error(self, key: str, error: Exception) -> None:
        with self._lock:
            self.failed_computations.value += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            completed = self.completed_computations.value
            avg_duration = (
                self.total_duration.value / completed if completed > 0 else 0.0
            )
            return {
                "total": self.total_computations.value,
                "completed": completed,
                "failed": self.failed_computations.value,
                "avg_duration": avg_duration,
            }


class ComputationBackend(ABC):
    """Abstract base for computation execution backends."""

    def __init__(self, middleware: Optional[List[Middleware]] = None):
        self.middleware = middleware or []

    @abstractmethod
    def launch(self, func: Callable, args: tuple, kwargs: dict, key: str) -> None:
        """
        Launch the computation asynchronously.

        The backend should execute func(*args, **kwargs) and store the result
        using the provided key.
        """
        pass

    def _run_middleware_before(
        self, func: Callable, args: tuple, kwargs: dict, key: str
    ) -> None:
        """Run all middleware before hooks."""
        for mw in self.middleware:
            try:
                mw.before_compute(func, args, kwargs, key)
            except Exception as e:
                logger.error(
                    f"Middleware {mw.__class__.__name__} before_compute failed: {e}"
                )

    def _run_middleware_after(self, key: str, result: ComputationResult) -> None:
        """Run all middleware after hooks."""
        for mw in self.middleware:
            try:
                mw.after_compute(key, result)
            except Exception as e:
                logger.error(
                    f"Middleware {mw.__class__.__name__} after_compute failed: {e}"
                )

    def _run_middleware_error(self, key: str, error: Exception) -> None:
        """Run all middleware error hooks."""
        for mw in self.middleware:
            try:
                mw.on_error(key, error)
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__} on_error failed: {e}")


def _worker_function(
    func: Callable,
    args: tuple,
    kwargs: dict,
    store: ComputationStore,
    key: str,
    middleware_configs: List[
        Dict[str, Any]
    ],  # Changed: pass configs instead of instances
) -> None:
    """Global worker function for multiprocessing with middleware support."""
    # Recreate middleware instances in the worker process
    middleware_instances = []
    for config in middleware_configs:
        cls = config["class"]
        kwargs = config.get("kwargs", {})
        middleware_instances.append(cls(**kwargs))

    # Run before hooks
    for mw in middleware_instances:
        try:
            mw.before_compute(func, args, kwargs, key)
        except Exception as e:
            logger.error(
                f"Middleware {mw.__class__.__name__} before_compute failed: {e}"
            )

    store[key] = ComputationResult(None, ComputationStatus.RUNNING)

    try:
        result_value = func(*args, **kwargs)
        result = ComputationResult(result_value, ComputationStatus.COMPLETED)
        store[key] = result

        # Run after hooks
        for mw in middleware_instances:
            try:
                mw.after_compute(key, result)
            except Exception as e:
                logger.error(
                    f"Middleware {mw.__class__.__name__} after_compute failed: {e}"
                )

    except Exception as e:
        result = ComputationResult(None, ComputationStatus.FAILED, error=e)
        store[key] = result

        # Run error hooks
        for mw in middleware_instances:
            try:
                mw.on_error(key, e)
            except Exception as ex:
                logger.error(
                    f"Middleware {mw.__class__.__name__} on_error failed: {ex}"
                )


class ProcessBackend(ComputationBackend):
    """Execute computations in separate processes with middleware support."""

    def __init__(
        self, store: ComputationStore, middleware: Optional[List[Middleware]] = None
    ):
        super().__init__(middleware)
        self.store = store
        self._ensure_fork_method()

    def _ensure_fork_method(self) -> None:
        """Ensure we're using fork method for better pickling compatibility."""
        try:
            current_method = multiprocessing.get_start_method()
            if current_method != "fork":
                multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            # Start method already set, which is fine
            pass

    def _serialize_middleware(self) -> List[Dict[str, Any]]:
        """Serialize middleware for worker process."""
        configs = []
        for mw in self.middleware:
            config = {"class": type(mw)}
            # Add specific configs for known middleware types
            if isinstance(mw, LoggingMiddleware):
                config["kwargs"] = {
                    "level": mw.level,
                    "logger_name": (
                        mw.logger.name if mw.logger.name != __name__ else None
                    ),
                }
            elif isinstance(mw, MetricsMiddleware):
                config["kwargs"] = {}
            configs.append(config)
        return configs

    def launch(self, func: Callable, args: tuple, kwargs: dict, key: str) -> None:
        """Launch computation in a new process."""
        self._run_middleware_before(func, args, kwargs, key)

        middleware_configs = self._serialize_middleware()
        process = multiprocessing.Process(
            target=_worker_function,
            args=(func, args, kwargs, self.store, key, middleware_configs),
        )
        process.start()


@dataclass
class ComputationHandle(Generic[T]):
    """
    Handle to check status and retrieve results of async computation.

    >>> handle = ComputationHandle("test-key", FileSystemStore("/tmp"))
    >>> handle.is_ready()
    False
    """

    key: str
    store: ComputationStore

    def is_ready(self) -> bool:
        """Check if the computation is complete."""
        return self.store[self.key].is_ready

    def get_status(self) -> ComputationStatus:
        """Get the current status of the computation."""
        return self.store[self.key].status

    def get_result(self, *, timeout: Optional[float] = None) -> T:
        """
        Get the computation result, optionally waiting for completion.

        Args:
            timeout: Maximum seconds to wait. None means no waiting.

        Raises:
            TimeoutError: If timeout expires before completion.
            Exception: If the computation failed.
        """

        def _poll_interval(elapsed: float) -> float:
            """Adaptive polling interval that increases with time."""
            if elapsed < 1:
                return 0.01
            elif elapsed < 10:
                return 0.1
            else:
                return 0.5

        start_time = time.time()

        while True:
            result = self.store[self.key]

            if result.status == ComputationStatus.COMPLETED:
                return result.value
            elif result.status == ComputationStatus.FAILED:
                raise result.error or Exception("Computation failed")

            if timeout is None:
                break

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Computation not ready after {timeout}s")

            time.sleep(_poll_interval(elapsed))

        raise RuntimeError("Result not ready and no timeout specified")

    def cancel(self) -> bool:
        """
        Attempt to cancel the computation.

        Returns:
            True if cancelled, False if already completed
        """
        result = self.store[self.key]
        if not result.is_ready:
            # Mark as failed with cancellation
            self.store[self.key] = ComputationResult(
                None, ComputationStatus.FAILED, error=Exception("Computation cancelled")
            )
            return True
        return False

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get computation metadata."""
        return self.store[self.key].metadata


def async_compute(
    backend: Optional[ComputationBackend] = None,
    store: Optional[ComputationStore] = None,
    *,
    base_path: str = "/tmp/computations",
    ttl_seconds: Optional[int] = 3600,
    serialization: SerializationFormat = SerializationFormat.JSON,
    middleware: Optional[List[Middleware]] = None,
) -> Callable:
    """
    Decorator to make functions asynchronous with status tracking.

    Args:
        backend: Backend to execute computations (default: ProcessBackend)
        store: Store for results (default: FileSystemStore)
        base_path: Path for FileSystemStore if no store provided
        ttl_seconds: Time-to-live for results (default: 1 hour)
        serialization: Format for storing results
        middleware: List of middleware components

    Example:
        >>> @async_compute(
        ...     ttl_seconds=7200,
        ...     middleware=[LoggingMiddleware(), MetricsMiddleware()]
        ... )
        ... def slow_computation(x: int) -> int:
        ...     return x * x
        >>> handle = slow_computation(5)  # Returns immediately
        >>> result = handle.get_result(timeout=10)
    """
    if store is None:
        store = FileSystemStore(
            base_path,
            ttl_seconds=ttl_seconds,
            serialization=serialization,  # This was missing - use the passed serialization
        )

    if backend is None:
        backend = ProcessBackend(store, middleware=middleware)

    def decorator(func: Callable[..., T]) -> Callable[..., ComputationHandle[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ComputationHandle[T]:
            key = store.create_key()
            backend.launch(func, args, kwargs, key)
            return ComputationHandle(key, store)

        # Attach utility methods
        wrapper.cleanup_expired = lambda: store.cleanup_expired()
        wrapper.store = store
        wrapper.backend = backend

        return wrapper

    return decorator


# Example implementations for other backends
class ThreadBackend(ComputationBackend):
    """
    Execute computations in separate threads (good for I/O-bound tasks).

    >>> import threading
    >>> store = FileSystemStore("/tmp/thread_computations")
    >>> backend = ThreadBackend(store)
    """

    def __init__(
        self, store: ComputationStore, middleware: Optional[List[Middleware]] = None
    ):
        super().__init__(middleware)
        self.store = store

    def launch(self, func: Callable, args: tuple, kwargs: dict, key: str) -> None:
        """Launch computation in a new thread."""
        import threading

        def _worker():
            self._run_middleware_before(func, args, kwargs, key)
            self.store[key] = ComputationResult(None, ComputationStatus.RUNNING)

            try:
                result = func(*args, **kwargs)
                final_result = ComputationResult(result, ComputationStatus.COMPLETED)
                self.store[key] = final_result
                self._run_middleware_after(key, final_result)
            except Exception as e:
                final_result = ComputationResult(
                    None, ComputationStatus.FAILED, error=e
                )
                self.store[key] = final_result
                self._run_middleware_error(key, e)

        thread = threading.Thread(target=_worker)
        thread.start()


class RemoteAPIBackend(ComputationBackend):
    """
    Example backend for remote computation services.

    This integrates with external APIs that have their own
    job tracking mechanisms.
    """

    def __init__(
        self,
        store: ComputationStore,
        *,
        api_url: str,
        api_key: str = "",
        middleware: Optional[List[Middleware]] = None,
    ):
        super().__init__(middleware)
        self.store = store
        self.api_url = api_url
        self.api_key = api_key

    def launch(self, func: Callable, args: tuple, kwargs: dict, key: str) -> None:
        """
        Launch computation via remote API.

        In a real implementation, this would:
        1. Serialize the function and arguments
        2. Submit to the remote API
        3. Poll for status updates
        4. Store results when complete
        """
        # This is a skeleton - actual implementation would depend on the API
        raise NotImplementedError("Implement based on your specific API")


# Context manager for temporary async computations
@contextmanager
def temporary_async_compute(**kwargs):
    """
    Context manager for temporary async computations with automatic cleanup.

    Example:
        >>> with temporary_async_compute(ttl_seconds=60) as async_func:
        ...     @async_func
        ...     def compute(x):
        ...         return x * 2
        ...     handle = compute(5)
        ...     result = handle.get_result(timeout=5)
    """
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    kwargs["base_path"] = temp_dir

    try:
        yield async_compute(**kwargs)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Example usage patterns
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Clean up any old files first
    import shutil
    import tempfile

    base_dir = "/tmp/computations_clean"
    if Path(base_dir).exists():
        shutil.rmtree(base_dir)

    # Create middleware
    logging_mw = LoggingMiddleware(level=logging.INFO)
    metrics_mw = MetricsMiddleware()

    # Example with middleware and TTL
    @async_compute(
        base_path=base_dir,
        ttl_seconds=300,  # 5 minutes
        middleware=[logging_mw, metrics_mw],
        serialization=SerializationFormat.PICKLE,  # For complex objects
    )
    def expensive_calculation(n: int) -> Dict[str, Any]:
        """Calculate factorial with metadata."""
        result = 1
        for i in range(1, n + 1):
            result *= i
            time.sleep(0.1)  # Simulate slow computation

        return {
            "factorial": result,
            "input": n,
            "timestamp": datetime.now().isoformat(),
        }

    # Launch computation
    handle = expensive_calculation(5)
    print(f"Computation launched with key: {handle.key}")

    # Check status
    print(f"Initial status: {handle.get_status()}")

    # Wait for result
    try:
        result = handle.get_result(timeout=10.0)  # Increased timeout
        print(f"Result: {result}")
    except TimeoutError:
        print("Computation still running...")

    # Check metrics (note: metrics won't be shared between processes)
    print(f"Main process metrics: {metrics_mw.get_stats()}")

    # Manual cleanup
    cleaned = expensive_calculation.cleanup_expired()
    print(f"Cleaned up {cleaned} expired results")
