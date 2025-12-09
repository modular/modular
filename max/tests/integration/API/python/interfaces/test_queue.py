# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import queue

from max.interfaces.queue import drain_queue


def test_drain_queue_with_integers() -> None:
    """Test that drain_queue correctly retrieves all integers added to a queue."""
    # Create a standard library queue
    test_queue = queue.Queue[int]()

    # Test with different numbers of integers
    test_cases = [0, 1, 5, 10, 100]

    for n in test_cases:
        # Clear the queue for each test case
        test_queue = queue.Queue[int]()

        # Create input data: list of integers from 0 to n-1
        input_integers = list(range(n))

        # Add all integers to the queue
        for i in input_integers:
            test_queue.put_nowait(i)

        # Drain the queue
        output_integers = drain_queue(test_queue)

        # Compare input and output
        assert output_integers == input_integers, (
            f"For n={n}: expected {input_integers}, got {output_integers}"
        )

        # Verify queue is empty after draining
        assert test_queue.empty(), (
            f"Queue should be empty after draining for n={n}"
        )


def test_drain_queue_with_custom_integers() -> None:
    """Test drain_queue with a custom sequence of integers."""
    test_queue = queue.Queue[int]()

    # Test with a custom sequence of integers (not just 0 to n-1)
    input_integers = [42, 7, 999, -5, 0, 123, 88]

    # Add all integers to the queue
    for i in input_integers:
        test_queue.put_nowait(i)

    # Drain the queue
    output_integers = drain_queue(test_queue)

    # Compare input and output (should preserve order)
    assert output_integers == input_integers, (
        f"Expected {input_integers}, got {output_integers}"
    )

    # Verify queue is empty after draining
    assert test_queue.empty(), "Queue should be empty after draining"


def test_drain_empty_queue() -> None:
    """Test that drain_queue returns empty list for empty queue."""
    test_queue = queue.Queue[int]()

    # Drain empty queue
    output_integers = drain_queue(test_queue)

    # Should return empty list
    assert output_integers == [], f"Expected empty list, got {output_integers}"

    # Queue should still be empty
    assert test_queue.empty(), "Queue should remain empty"
