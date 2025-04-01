# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for running functions in an isolated process."""

import multiprocessing
import sys
import time
from pickle import PicklingError
from queue import Empty
from traceback import format_exc
from typing import Callable, TypeVar

ReturnT = TypeVar("ReturnT")


def _isolated_process_wrapper(
    func: Callable[[], ReturnT], result_queue: multiprocessing.Queue
) -> None:
    """Execute a function and queue the result or exception.

    The result is put in the queue as (True, result) on success or
    (False, (exception, traceback_string)) on failure to preserve both
    the exception object and its traceback information.

    Args:
        func: A callable that takes no arguments and returns a value of type ReturnT.
        result_queue: A Queue to store the (success, payload) tuple.
    """
    try:
        result = func()
        result_queue.put((True, result))
    except Exception as e:
        tb_str = format_exc()
        try:
            # Try to pickle the exception and traceback
            result_queue.put((False, (e, tb_str)))
        except (TypeError, PicklingError):
            # If exception can't be pickled, use a string representation
            result_queue.put(
                (
                    False,
                    (RuntimeError(f"Unpicklable exception: {str(e)}"), tb_str),
                )
            )


def run_in_isolated_process(
    func: Callable[[], ReturnT],
    timeout: int = 600,
    print_traceback: bool = True,
) -> ReturnT:
    """
    Execute a function in an isolated process with timeout handling.

    Args:
        func: The function to execute (with arguments already bound)
        timeout: Timeout in seconds (default: 10 minutes)
        print_traceback: Whether to print exception traceback information (default: True)

    Returns:
        The return value from the function, with the same type as func's return type
    """
    # Needed for multiprocessing to work nicely with CUDA.
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(maxsize=1)

    process = ctx.Process(target=_isolated_process_wrapper, args=(func, queue))

    process.start()
    process.join(timeout)

    try:
        # Process should not be alive if it has finished execution,
        # this means it's still running after the timeout
        if process.is_alive():
            print("Reached timeout, subprocess still running", file=sys.stderr)
            print("Attempting to terminate process gracefully", file=sys.stderr)
            process.terminate()
            time.sleep(5)
            if process.is_alive():
                print(
                    "No response after 5 seconds, killing process forcefully",
                    file=sys.stderr,
                )
                process.kill()
                time.sleep(2)  # Provide time for process to be killed
            raise multiprocessing.TimeoutError(
                f"Function execution timed out after {timeout} seconds"
            )
        try:
            success, payload = queue.get(timeout=2)
        except Empty as e:
            raise RuntimeError(
                "Subprocess did not return a result. It may have crashed."
            ) from e
        if not success:
            exception, tb_str = payload
            if print_traceback:
                print(
                    f"Exception caught in child process: {exception}",
                    file=sys.stderr,
                )
                print("\nFull traceback from child process:", file=sys.stderr)
                print(tb_str, file=sys.stderr)
            raise exception

        return payload
    finally:
        process.close()
        queue.close()
