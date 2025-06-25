"""
Tests for strand.taskrunning.utils.run_process
"""

import time
from strand.taskrunning.utils import run_process


# --- Child process functions must be at module level for multiprocessing ---
def child_basic():
    print("Child process started")
    time.sleep(2)
    print("Child process exiting")


def child_long():
    print("Long child process started")
    while True:
        time.sleep(0.1)


# --- Tests ---
def test_run_process_basic():
    """Test that run_process launches and kills a process."""
    with run_process(
        child_basic, process_name="test_child", is_ready=0.2, timeout=5
    ) as proc:
        assert proc is not None
        assert proc.is_alive()
        # Wait a bit to ensure process is running
        time.sleep(0.5)
        assert proc.is_alive()
    # After context, process should be dead
    assert not proc.is_alive()


def test_run_process_already_running():
    """Test that run_process does not launch if process_already_running returns True."""

    def dummy_child():
        print("Should not be called!")

    called = []

    def already_running():
        called.append(True)
        return True

    with run_process(dummy_child, process_already_running=already_running) as proc:
        assert proc is None
    assert called, "process_already_running should have been called"


def test_run_process_force_kill():
    """Test that run_process force kills a long-running process."""
    with run_process(
        child_long, process_name="long_child", is_ready=0.2, timeout=5, force_kill=True
    ) as proc:
        assert proc is not None
        assert proc.is_alive()
        time.sleep(0.5)
    assert not proc.is_alive()


if __name__ == "__main__":
    test_run_process_basic()
    test_run_process_already_running()
    test_run_process_force_kill()
    print("All run_process tests passed.")
