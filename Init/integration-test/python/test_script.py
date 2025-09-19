#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import os
import sys

# Add the current directory to Python path to find init_bindings
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

try:
    import init_bindings
except ImportError as e:
    print(f"Failed to import init_bindings: {e}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    sys.exit(1)


def deep_function_3():
    """Third level function that triggers the segfault."""
    print(
        "About to trigger SIGSEGV from Python context via C++", file=sys.stderr
    )
    # Trigger a segfault from C++ (which was called from Python context)
    init_bindings.trigger_segfault_from_cpp()


def deep_function_2():
    """Second level function in the call stack."""
    deep_function_3()


def deep_function_1():
    """First level function in the call stack."""
    deep_function_2()


def main():
    """Main function that sets up the signal handler and triggers the crash."""
    print("Starting signal handler test", file=sys.stderr)

    # Initialize the C++ signal handler
    init_bindings.initialize_signal_handler("test_python_signal_handler")
    print("Signal handler initialized", file=sys.stderr)

    # Call the function chain that will trigger the signal
    deep_function_1()


if __name__ == "__main__":
    main()
