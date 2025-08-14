# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from gpu.host import DeviceContext, DeviceStream
from testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_raises,
)


def test_stream_priority_range(ctx: DeviceContext):
    print("Test that stream_priority_range() returns a valid priority range.")
    var priority_range = ctx.stream_priority_range()

    # The range should be valid (least >= greatest)
    # generally largest is -5 while least is 0
    assert_true(
        priority_range.least >= priority_range.greatest,
        "Priority range should have least >= greatest",
    )


def test_create_stream_default(ctx: DeviceContext):
    print("Test creating a stream with default parameters (blocking=True).")
    var stream = ctx.create_stream()

    # Basic operations should work with the stream
    stream.synchronize()


def test_create_stream_with_priority(ctx: DeviceContext):
    print("Test creating streams with different priority values.")
    var priority_range = ctx.stream_priority_range()

    # Test with lowest priority
    var low_priority_stream = ctx.create_stream(
        priority=priority_range.least, blocking=True
    )
    low_priority_stream.synchronize()

    # Test with highest priority
    var high_priority_stream = ctx.create_stream(
        priority=priority_range.greatest, blocking=True
    )
    high_priority_stream.synchronize()

    # Test with middle priority (if range allows)
    if priority_range.least < priority_range.greatest:
        var mid_priority = (priority_range.least + priority_range.greatest) // 2
        var mid_priority_stream = ctx.create_stream(
            priority=mid_priority, blocking=True
        )
        mid_priority_stream.synchronize()


def test_multiple_priority_streams(ctx: DeviceContext):
    print("Test creating multiple streams with different priorities.")
    var priority_range = ctx.stream_priority_range()

    # Create multiple streams with different priorities
    var streams = List[DeviceStream]()

    # Create streams with lowest and highest priority
    streams.append(ctx.create_stream(priority=priority_range.least))
    streams.append(ctx.create_stream(priority=priority_range.greatest))

    # If we have a range of priorities, create some intermediate ones
    if priority_range.greatest > priority_range.least:
        var step = max(1, (priority_range.greatest - priority_range.least) // 4)
        var current_priority = priority_range.least + step
        while current_priority < priority_range.greatest:
            streams.append(ctx.create_stream(priority=current_priority))
            current_priority += step

    # Synchronize all streams
    for i in range(len(streams)):
        streams[i].synchronize()


def test_stream_blocking_and_non_blocking(ctx: DeviceContext):
    print("Test edge cases for stream priority creation.")
    var priority_range = ctx.stream_priority_range()

    # Test creating multiple streams with the same priority
    var stream1 = ctx.create_stream(priority=priority_range.least)
    var stream2 = ctx.create_stream(priority=priority_range.least)

    stream1.synchronize()
    stream2.synchronize()

    # Test mixing blocking and non-blocking with same priority
    var blocking_stream = ctx.create_stream(
        priority=priority_range.greatest, blocking=True
    )
    var non_blocking_stream = ctx.create_stream(
        priority=priority_range.greatest, blocking=False
    )

    blocking_stream.synchronize()
    non_blocking_stream.synchronize()


def main():
    with DeviceContext() as ctx:
        test_stream_priority_range(ctx)
        test_create_stream_default(ctx)
        test_create_stream_with_priority(ctx)
        test_multiple_priority_streams(ctx)
        test_stream_blocking_and_non_blocking(ctx)
