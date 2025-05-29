"""Test au functionality"""

import pytest
import tempfile
import shutil
import time
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strand.au import (
    async_compute,
    ComputationStatus,
    ComputationResult,
    ComputationHandle,
    SerializationFormat,
    FileSystemStore,
    ProcessBackend,
    ThreadBackend,
    LoggingMiddleware,
    MetricsMiddleware,
    temporary_async_compute,
)


class TestComputationResult:
    """Test ComputationResult dataclass."""

    def test_is_ready_completed(self):
        result = ComputationResult("value", ComputationStatus.COMPLETED)
        assert result.is_ready is True

    def test_is_ready_failed(self):
        result = ComputationResult(None, ComputationStatus.FAILED)
        assert result.is_ready is True

    def test_is_ready_pending(self):
        result = ComputationResult(None, ComputationStatus.PENDING)
        assert result.is_ready is False

    def test_is_ready_running(self):
        result = ComputationResult(None, ComputationStatus.RUNNING)
        assert result.is_ready is False

    def test_duration_calculation(self):
        start_time = datetime.now()
        result = ComputationResult(
            "value", ComputationStatus.COMPLETED, created_at=start_time
        )
        result.completed_at = start_time + timedelta(seconds=5)
        assert result.duration.total_seconds() == 5.0

    def test_duration_none_when_not_completed(self):
        result = ComputationResult("value", ComputationStatus.RUNNING)
        assert result.duration is None


class TestFileSystemStore:
    """Test FileSystemStore functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_store_creation(self, temp_dir):
        store = FileSystemStore(temp_dir)
        assert store.base_path == Path(temp_dir)
        assert store.base_path.exists()

    def test_create_key(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key1 = store.create_key()
        key2 = store.create_key()
        assert key1 != key2
        assert isinstance(key1, str)
        assert isinstance(key2, str)

    def test_store_and_retrieve_result_json(self, temp_dir):
        store = FileSystemStore(temp_dir, serialization=SerializationFormat.JSON)
        key = store.create_key()
        result = ComputationResult(42, ComputationStatus.COMPLETED)

        store[key] = result
        retrieved = store[key]

        assert retrieved.value == 42
        assert retrieved.status == ComputationStatus.COMPLETED
        assert retrieved.is_ready

    def test_store_and_retrieve_result_pickle(self, temp_dir):
        store = FileSystemStore(temp_dir, serialization=SerializationFormat.PICKLE)
        key = store.create_key()
        complex_data = {"list": [1, 2, 3], "nested": {"key": "value"}}
        result = ComputationResult(complex_data, ComputationStatus.COMPLETED)

        store[key] = result
        retrieved = store[key]

        assert retrieved.value == complex_data
        assert retrieved.status == ComputationStatus.COMPLETED

    def test_nonexistent_key_returns_pending(self, temp_dir):
        store = FileSystemStore(temp_dir)
        result = store["nonexistent-key"]
        assert result.status == ComputationStatus.PENDING
        assert result.value is None

    def test_delete_result(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = store.create_key()
        result = ComputationResult("test", ComputationStatus.COMPLETED)

        store[key] = result
        assert store[key].value == "test"

        del store[key]
        retrieved = store[key]
        assert retrieved.status == ComputationStatus.PENDING

    def test_iteration(self, temp_dir):
        store = FileSystemStore(temp_dir)
        keys = [store.create_key() for _ in range(3)]

        for key in keys:
            store[key] = ComputationResult(f"value-{key}", ComputationStatus.COMPLETED)

        stored_keys = list(store)
        assert len(stored_keys) == 3
        for key in keys:
            assert key in stored_keys

    def test_length(self, temp_dir):
        store = FileSystemStore(temp_dir)
        assert len(store) == 0

        keys = [store.create_key() for _ in range(5)]
        for key in keys:
            store[key] = ComputationResult("value", ComputationStatus.COMPLETED)

        assert len(store) == 5

    def test_ttl_expiration(self, temp_dir):
        store = FileSystemStore(temp_dir, ttl_seconds=1)

        # Create an old result
        old_time = datetime.now() - timedelta(seconds=2)
        old_result = ComputationResult(
            "old", ComputationStatus.COMPLETED, created_at=old_time
        )
        old_result.completed_at = old_time

        # Create a fresh result
        fresh_result = ComputationResult("fresh", ComputationStatus.COMPLETED)

        assert store.is_expired(old_result) is True
        assert store.is_expired(fresh_result) is False

    def test_cleanup_expired(self, temp_dir):
        store = FileSystemStore(temp_dir, ttl_seconds=1)

        # Create some results
        keys = [store.create_key() for _ in range(3)]

        # Make first result expired - ensure it's marked as completed
        old_time = datetime.now() - timedelta(seconds=2)
        old_result = ComputationResult(
            "old", ComputationStatus.COMPLETED, created_at=old_time
        )
        old_result.completed_at = old_time  # Mark as completed so TTL applies
        store[keys[0]] = old_result

        # Fresh results - also mark as completed
        for key in keys[1:]:
            fresh_result = ComputationResult("fresh", ComputationStatus.COMPLETED)
            fresh_result.completed_at = datetime.now()  # Mark as completed
            store[key] = fresh_result

        # Cleanup should remove 1 expired result
        removed = store.cleanup_expired()
        assert removed == 1
        assert len(store) == 2


class TestComputationHandle:
    """Test ComputationHandle functionality."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_handle_creation(self, temp_dir):
        store = FileSystemStore(temp_dir)
        handle = ComputationHandle("test-key", store)
        assert handle.key == "test-key"
        assert handle.store is store

    def test_is_ready_pending(self, temp_dir):
        store = FileSystemStore(temp_dir)
        handle = ComputationHandle("nonexistent", store)
        assert handle.is_ready() is False

    def test_is_ready_completed(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = "test-key"
        store[key] = ComputationResult("done", ComputationStatus.COMPLETED)

        handle = ComputationHandle(key, store)
        assert handle.is_ready() is True

    def test_get_status(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = "test-key"
        store[key] = ComputationResult(None, ComputationStatus.RUNNING)

        handle = ComputationHandle(key, store)
        assert handle.get_status() == ComputationStatus.RUNNING

    def test_get_result_immediate(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = "test-key"
        store[key] = ComputationResult(42, ComputationStatus.COMPLETED)

        handle = ComputationHandle(key, store)
        result = handle.get_result()
        assert result == 42

    def test_get_result_with_timeout_success(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = "test-key"

        # Start with pending
        handle = ComputationHandle(key, store)

        # Simulate completion after a short delay
        def complete_after_delay():
            time.sleep(0.1)
            store[key] = ComputationResult("completed", ComputationStatus.COMPLETED)

        import threading

        thread = threading.Thread(target=complete_after_delay)
        thread.start()

        result = handle.get_result(timeout=1.0)
        assert result == "completed"
        thread.join()

    def test_get_result_timeout_error(self, temp_dir):
        store = FileSystemStore(temp_dir)
        handle = ComputationHandle("nonexistent", store)

        with pytest.raises(TimeoutError):
            handle.get_result(timeout=0.1)

    def test_get_result_computation_failed(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = "test-key"
        error = ValueError("test error")
        store[key] = ComputationResult(None, ComputationStatus.FAILED, error=error)

        handle = ComputationHandle(key, store)
        with pytest.raises(ValueError, match="test error"):
            handle.get_result()

    def test_get_result_no_timeout_not_ready(self, temp_dir):
        store = FileSystemStore(temp_dir)
        handle = ComputationHandle("nonexistent", store)

        with pytest.raises(RuntimeError, match="Result not ready"):
            handle.get_result()

    def test_cancel_pending_computation(self, temp_dir):
        store = FileSystemStore(temp_dir)
        handle = ComputationHandle("test-key", store)

        success = handle.cancel()
        assert success is True
        assert handle.get_status() == ComputationStatus.FAILED

    def test_cancel_completed_computation(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = "test-key"
        store[key] = ComputationResult("done", ComputationStatus.COMPLETED)

        handle = ComputationHandle(key, store)
        success = handle.cancel()
        assert success is False

    def test_metadata_access(self, temp_dir):
        store = FileSystemStore(temp_dir)
        key = "test-key"
        metadata = {"user": "test", "priority": "high"}
        result = ComputationResult(
            "value", ComputationStatus.COMPLETED, metadata=metadata
        )
        store[key] = result

        handle = ComputationHandle(key, store)
        assert handle.metadata == metadata


class TestMiddleware:
    """Test middleware functionality."""

    def test_logging_middleware(self):
        middleware = LoggingMiddleware()

        # Test before_compute
        func = lambda x: x * 2
        middleware.before_compute(func, (5,), {}, "test-key")

        # Test after_compute
        result = ComputationResult(10, ComputationStatus.COMPLETED)
        middleware.after_compute("test-key", result)

        # Test on_error
        error = ValueError("test error")
        middleware.on_error("test-key", error)

    def test_metrics_middleware(self):
        middleware = MetricsMiddleware()

        # Initial stats
        stats = middleware.get_stats()
        assert stats["total"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
        assert stats["avg_duration"] == 0.0

        # Simulate computation
        func = lambda x: x * 2
        middleware.before_compute(func, (5,), {}, "key1")

        # Complete it
        result = ComputationResult(10, ComputationStatus.COMPLETED)
        result.completed_at = result.created_at + timedelta(seconds=2)
        middleware.after_compute("key1", result)

        # Check updated stats
        stats = middleware.get_stats()
        assert stats["total"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 0
        assert stats["avg_duration"] == 2.0

        # Simulate error
        middleware.before_compute(func, (5,), {}, "key2")
        middleware.on_error("key2", ValueError("error"))

        stats = middleware.get_stats()
        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["failed"] == 1


class TestAsyncComputeDecorator:
    """Test the main async_compute decorator."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_basic_decoration(self, temp_dir):
        @async_compute(base_path=temp_dir)
        def simple_func(x):
            return x * 2

        # Function should return a handle
        handle = simple_func(21)
        assert isinstance(handle, ComputationHandle)
        assert isinstance(handle.key, str)

    def test_computation_with_thread_backend(self, temp_dir):
        store = FileSystemStore(temp_dir)
        backend = ThreadBackend(store)

        @async_compute(backend=backend, store=store)
        def thread_func(x):
            time.sleep(0.1)  # Small delay to ensure threading
            return x**2

        handle = thread_func(5)
        result = handle.get_result(timeout=2.0)
        assert result == 25

    def test_computation_with_middleware(self, temp_dir):
        metrics = MetricsMiddleware()

        @async_compute(base_path=temp_dir, middleware=[metrics])
        def monitored_func(x):
            return x + 1

        handle = monitored_func(10)
        result = handle.get_result(timeout=5.0)
        assert result == 11

        # Check metrics were recorded
        stats = metrics.get_stats()
        assert stats["total"] >= 1

    def test_function_with_error(self, temp_dir):
        @async_compute(base_path=temp_dir)
        def failing_func(x):
            if x < 0:
                raise ValueError("x must be positive")
            return x

        handle = failing_func(-5)

        # Wait for computation to complete and fail
        time.sleep(0.5)
        assert handle.get_status() == ComputationStatus.FAILED

        with pytest.raises(ValueError, match="x must be positive"):
            handle.get_result()

    def test_attached_utility_methods(self, temp_dir):
        @async_compute(base_path=temp_dir, ttl_seconds=1)
        def utility_func(x):
            return x

        # Test that utility methods are attached
        assert hasattr(utility_func, 'cleanup_expired')
        assert hasattr(utility_func, 'store')
        assert hasattr(utility_func, 'backend')

        # Test cleanup method works
        removed = utility_func.cleanup_expired()
        assert isinstance(removed, int)

    def test_custom_serialization(self, temp_dir):
        @async_compute(base_path=temp_dir, serialization=SerializationFormat.PICKLE)
        def complex_data_func():
            return {"complex": [1, 2, {"nested": True}]}

        handle = complex_data_func()
        result = handle.get_result(timeout=5.0)
        expected = {"complex": [1, 2, {"nested": True}]}
        assert result == expected

    def test_ttl_configuration(self, temp_dir):
        @async_compute(base_path=temp_dir, ttl_seconds=3600)
        def ttl_func(x):
            return x

        # Check that store has correct TTL
        assert ttl_func.store.ttl_seconds == 3600


class TestTemporaryAsyncCompute:
    """Test temporary_async_compute context manager."""

    def test_temporary_context_manager(self):
        with temporary_async_compute(ttl_seconds=60) as async_func:

            @async_func
            def temp_func(x):
                return x * 3

            handle = temp_func(7)
            result = handle.get_result(timeout=5.0)
            assert result == 21

        # Directory should be cleaned up after context exit
        # We can't easily test this without knowing the temp directory path

    def test_temporary_with_custom_config(self):
        with temporary_async_compute(
            ttl_seconds=120, serialization=SerializationFormat.PICKLE
        ) as async_func:

            @async_func
            def config_func():
                return {"test": "data"}

            handle = config_func()
            result = handle.get_result(timeout=5.0)
            assert result == {"test": "data"}


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_end_to_end_workflow(self, temp_dir):
        """Test a complete workflow with multiple computations."""
        metrics = MetricsMiddleware()

        @async_compute(
            base_path=temp_dir,
            ttl_seconds=3600,
            middleware=[metrics],
            serialization=SerializationFormat.PICKLE,
        )
        def process_data(data):
            # Simulate some processing time
            time.sleep(0.1)
            return {"processed": data, "timestamp": time.time()}

        # Launch multiple computations
        handles = []
        for i in range(3):
            handle = process_data(f"data-{i}")
            handles.append(handle)

        # Collect results
        results = []
        for handle in handles:
            result = handle.get_result(timeout=5.0)
            results.append(result)
            assert handle.is_ready()
            assert handle.get_status() == ComputationStatus.COMPLETED

        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["processed"] == f"data-{i}"
            assert "timestamp" in result

        # Check metrics
        stats = metrics.get_stats()
        assert stats["total"] >= 3
        assert stats["failed"] == 0

    def test_shared_infrastructure(self, temp_dir):
        """Test multiple functions sharing the same backend and store."""
        store = FileSystemStore(temp_dir, ttl_seconds=7200)
        backend = ThreadBackend(store)  # Use threads for faster testing

        @async_compute(backend=backend, store=store)
        def step1(x):
            return x + 10

        @async_compute(backend=backend, store=store)
        def step2(x):
            return x * 2

        # Chain computations
        h1 = step1(5)
        intermediate = h1.get_result(timeout=2.0)
        assert intermediate == 15

        h2 = step2(intermediate)
        final = h2.get_result(timeout=2.0)
        assert final == 30

        # Verify both used the same store
        assert step1.store is step2.store
        assert step1.backend is step2.backend

    def test_error_handling_and_recovery(self, temp_dir):
        """Test error handling in various scenarios."""

        @async_compute(base_path=temp_dir)
        def maybe_fail(x, should_fail=False):
            if should_fail:
                raise RuntimeError(f"Intentional failure for {x}")
            return x * 2

        # Successful computation
        success_handle = maybe_fail(5, should_fail=False)
        result = success_handle.get_result(timeout=5.0)
        assert result == 10

        # Failed computation
        fail_handle = maybe_fail(3, should_fail=True)
        time.sleep(0.5)  # Let it fail
        assert fail_handle.get_status() == ComputationStatus.FAILED

        with pytest.raises(RuntimeError, match="Intentional failure for 3"):
            fail_handle.get_result()

        # Cancellation
        cancel_handle = maybe_fail(7, should_fail=False)
        cancelled = cancel_handle.cancel()
        assert cancelled is True
        assert cancel_handle.get_status() == ComputationStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__])
