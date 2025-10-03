# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import faulthandler
import signal
import sys

import init_bindings


def deep_function_3() -> None:
    """Third level function that triggers the segfault."""
    print(
        "About to trigger SIGSEGV from Python context via C++", file=sys.stderr
    )
    # Trigger a segfault from C++ (which was called from Python context)
    init_bindings.trigger_segfault_from_cpp()


def deep_function_2() -> None:
    """Second level function in the call stack."""
    deep_function_3()


def deep_function_1() -> None:
    """First level function in the call stack."""
    deep_function_2()


def main() -> None:
    """Main function that sets up the signal handler and triggers the crash."""
    print("Starting signal handler test", file=sys.stderr)

    # Register faulthandler for SIGUSR2 to enable async-safe Python stack traces
    # Note: chain=False so it doesn't interfere with our signal handling flow
    faulthandler.register(
        signal.SIGUSR2, file=sys.stderr, all_threads=True, chain=False
    )
    print("Python faulthandler registered for SIGUSR2", file=sys.stderr)

    # Initialize the C++ signal handler
    init_bindings.initialize_signal_handler("test_python_signal_handler")
    print("Signal handler initialized", file=sys.stderr)

    # Call the function chain that will trigger the signal
    deep_function_1()


if __name__ == "__main__":
    main()
