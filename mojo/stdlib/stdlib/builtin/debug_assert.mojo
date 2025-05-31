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
"""Implements run-time assertions.

These are Mojo built-ins, so you don't need to import them.
"""


from os import abort
from sys import is_amd_gpu, is_gpu, is_nvidia_gpu, llvm_intrinsic
from sys._build import is_debug_build
from sys.ffi import c_char, c_size_t, c_uint, external_call
from sys.intrinsics import block_idx, thread_idx
from sys.param_env import env_get_string

from builtin._location import __call_location, _SourceLocation

from utils.write import WritableVariadicPack

alias ASSERT_MODE = env_get_string["ASSERT", "safe"]()
alias WRITE_MODE = Int
alias WRITE_MODE_REG = 0
alias WRITE_MODE_MEM = 1


@no_inline
fn _assert_enabled[assert_mode: StaticString, cpu_only: Bool]() -> Bool:
    constrained[
        ASSERT_MODE == "none"
        or ASSERT_MODE == "warn"
        or ASSERT_MODE == "safe"
        or ASSERT_MODE == "all",
        "-D ASSERT=",
        ASSERT_MODE,
        " but must be one of: none, warn, safe, all",
    ]()
    constrained[
        assert_mode == "none" or assert_mode == "safe",
        "assert_mode=",
        assert_mode,
        " but must be one of: none, safe",
    ]()

    @parameter
    if ASSERT_MODE == "none" or (is_gpu() and cpu_only):
        return False
    elif ASSERT_MODE == "all" or ASSERT_MODE == "warn" or is_debug_build():
        return True
    else:
        return ASSERT_MODE == assert_mode


@always_inline
fn debug_assert[
    cond: fn () capturing [_] -> Bool,
    write_mode: WRITE_MODE = WRITE_MODE_REG,
    assert_mode: StaticString = "none",
    cpu_only: Bool = False,
    *Ts: Writable,
](*messages: *Ts, location: Optional[_SourceLocation] = None):
    """Asserts that the condition is true at run time.

    If the condition is false, the assertion displays the given message and
    causes the program to exit.

    You can pass in multiple arguments to generate a formatted
    message. No string allocation occurs unless the assertion is triggered.

    ```mojo
    x = 0
    debug_assert(x > 0, "expected x to be more than 0 but got: ", x)
    ```

    Normal assertions are off by default—they only run when the program is
    compiled with all assertions enabled. You can set the `assert_mode` to
    `safe` to create an assertion that's on by default:

    ```mojo
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    Use the `ASSERT` variable to turn assertions on or off when building or
    running a Mojo program:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    The `ASSERT` variable takes the following values:

    - all: Turn on all assertions.
    - safe: Turn on "safe" assertions only. This is the default.
    - none: Turn off all assertions, for performance at the cost of safety.
    - warn: Turn on all assertions, but print any errors instead of exiting.

    To ensure that you have no run-time penalty from your assertions even when
    they're disabled, make sure there are no side effects in your message and
    condition expressions. For example:

    ```mojo
    person = "name: john, age: 50"
    name = "john"
    debug_assert(String("name: ") + name == person, "unexpected name")
    ```

    This will have a run-time penalty due to allocating a `String` in the
    condition expression, even when assertions are disabled. To avoid this, put
    the condition inside a closure so it runs only when the assertion is turned
    on:

    ```mojo
    fn check_name() capturing -> Bool:
        return String("name: ") + name == person

    debug_assert[check_name]("unexpected name")
    ```

    If you need to allocate, and so don't want the assert to ever run on GPU,
    you can set it to CPU only:

    ```mojo
    debug_assert[check_name, cpu_only=True]("unexpected name")
    ```

    For compile-time assertions, see
    [`constrained()`](/mojo/stdlib/builtin/constrained/constrained).

    Parameters:
        cond: The function to invoke to check if the assertion holds.
        write_mode: Determines whether to keep values in register or not.
        assert_mode: Determines when the assert is turned on.
            - default ("none"): Turned on when compiled with `-D ASSERT=all`.
            - "safe": Turned on by default.
        cpu_only: If true, only run the assert on CPU.
        Ts: The element types for the message arguments.

    Args:
        messages: A set of [`Writable`](/mojo/stdlib/utils/write/Writable/)
            arguments to convert to a `String` message.
        location: The location of the error (defaults to `__call_location`).
    """

    @parameter
    if _assert_enabled[assert_mode, cpu_only]():
        if cond():
            return

        # TODO(KERN-1738): Resolve stack usage for AMDGPU target.
        @parameter
        if write_mode == WRITE_MODE_MEM and not (is_gpu() and is_amd_gpu()):
            var str: String = ""

            @parameter
            for i in range(messages.__len__()):
                messages[i].write_to(str)
            _debug_assert_msg_mem(location.or_else(__call_location()), str)
        else:
            _debug_assert_msg(messages, location.or_else(__call_location()))


@always_inline
fn debug_assert[
    write_mode: WRITE_MODE = WRITE_MODE_REG,
    assert_mode: StaticString = "none",
    cpu_only: Bool = False,
    *Ts: Writable,
](cond: Bool, *messages: *Ts, location: Optional[_SourceLocation] = None):
    """Asserts that the condition is true at run time.

    If the condition is false, the assertion displays the given message and
    causes the program to exit.

    You can pass in multiple arguments to generate a formatted
    message. No string allocation occurs unless the assertion is triggered.

    ```mojo
    x = 0
    debug_assert(x > 0, "expected x to be more than 0 but got: ", x)
    ```

    Normal assertions are off by default—they only run when the program is
    compiled with all assertions enabled. You can set the `assert_mode` to
    `safe` to create an assertion that's on by default:

    ```mojo
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    Use the `ASSERT` variable to turn assertions on or off when building or
    running a Mojo program:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    The `ASSERT` variable takes the following values:

    - all: Turn on all assertions.
    - safe: Turn on "safe" assertions only. This is the default.
    - none: Turn off all assertions, for performance at the cost of safety.
    - warn: Turn on all assertions, but print any errors instead of exiting.

    To ensure that you have no run-time penalty from your assertions even when
    they're disabled, make sure there are no side effects in your message and
    condition expressions. For example:

    ```mojo
    person = "name: john, age: 50"
    name = "john"
    debug_assert(String("name: ") + name == person, "unexpected name")
    ```

    This will have a run-time penalty due to allocating a `String` in the
    condition expression, even when assertions are disabled. To avoid this, put
    the condition inside a closure so it runs only when the assertion is turned
    on:

    ```mojo
    fn check_name() capturing -> Bool:
        return String("name: ") + name == person

    debug_assert[check_name]("unexpected name")
    ```

    If you need to allocate, and so don't want the assert to ever run on GPU,
    you can set it to CPU only:

    ```mojo
    debug_assert[check_name, cpu_only=True]("unexpected name")
    ```

    For compile-time assertions, see
    [`constrained()`](/mojo/stdlib/builtin/constrained/constrained).

    Parameters:
        write_mode: Determines whether to keep values in register or not.
        assert_mode: Determines when the assert is turned on.
            - default ("none"): Turned on when compiled with `-D ASSERT=all`.
            - "safe": Turned on by default.
        cpu_only: If true, only run the assert on CPU.
        Ts: The element types for the message arguments.

    Args:
        cond: The bool value to assert.
        messages: A set of [`Writable`](/mojo/stdlib/utils/write/Writable/)
            arguments to convert to a `String` message.
        location: The location of the error (defaults to `__call_location`).
    """

    @parameter
    if _assert_enabled[assert_mode, cpu_only]():
        if cond:
            return

        # TODO(KERN-1738): Resolve stack usage for AMDGPU target.
        @parameter
        if write_mode == WRITE_MODE_MEM and not (is_gpu() and is_amd_gpu()):
            var str: String = ""

            @parameter
            for i in range(messages.__len__()):
                messages[i].write_to(str)
            _debug_assert_msg_mem(location.or_else(__call_location()), str)
        else:
            _debug_assert_msg(messages, location.or_else(__call_location()))


@no_inline
fn _debug_assert_msg_mem(loc: _SourceLocation, message: String):
    """This version of debug assert is needed to enable a register allocator optimization.
    """

    alias kind_str = StaticString(
        "Warning: "
    ) if ASSERT_MODE == "warn" else "Error: "

    @parameter
    if is_gpu():

        @parameter
        if is_amd_gpu():
            # TODO(KERN-1738): Resolve stack usage for AMDGPU target.
            print("Assert failed")
        else:
            var str: String = ""
            var info = _GPUThreadInfo()
            str.write("At", loc, ": ", info, " Assert ", kind_str, message)
            print(str)
    else:
        var str: String = "At " + loc.__str__() + ": Assert " + kind_str + message
        print(str)

    @parameter
    if ASSERT_MODE != "warn":
        abort()


@fieldwise_init
struct _GPUThreadInfo(Writable):
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "block: [",
            block_idx.x,
            ",",
            block_idx.y,
            ",",
            block_idx.z,
            "] thread: [",
            thread_idx.x,
            ",",
            thread_idx.y,
            ",",
            thread_idx.z,
            "]",
        )


@no_inline
fn _debug_assert_msg[
    origin: ImmutableOrigin, *Ts: Writable
](messages: VariadicPack[_, origin, Writable, *Ts], loc: _SourceLocation):
    """Aborts with (or prints) the given message and location.

    This function is intentionally marked as no_inline to reduce binary size.

    Note that it's important that this function doesn't get inlined; otherwise,
    an indirect recursion of @always_inline functions is possible (e.g. because
    abort's implementation could use debug_assert)
    """

    alias kind_str = StaticString(
        "Warning: "
    ) if ASSERT_MODE == "warn" else "Error: "

    @parameter
    if is_gpu():

        @parameter
        if is_amd_gpu():
            # TODO(KERN-1738): Resolve stack usage for AMDGPU target.
            print("Assert failed")
        else:
            print(
                "At ",
                loc,
                ": ",
                _GPUThreadInfo(),
                " Assert ",
                kind_str,
                WritableVariadicPack(messages),
                sep="",
            )
    else:
        print(
            "At ",
            loc,
            ": ",
            " Assert ",
            kind_str,
            WritableVariadicPack(messages),
            sep="",
        )

    @parameter
    if ASSERT_MODE != "warn":
        abort()
