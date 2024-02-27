# Mojo Debug Tests

There are two main ways to test LLDB's ability to parse DWARF and other debug
info sections and provide a nice debugging experience: llvm-lit tests and python
SB tests.

See `Support/python/modular/utils/debuglib/debugger.py` for some useful
information regarding the underlying LLDB support.

## llvm-lit tests

These tests assert on the LLDB CLI's output and should print variables using
the `frame variable` (aka `v`) command. `expr` (aka `p`) should be avoided
because it often tries to JIT, which we don't support yet.

An advantage of `frame variable` is that it allows static dereferencing of
pointers with the `*` operator, e.g. `frame variable *a_pointer`.

Besides that, it also allows variable paths, e.g.
`frame variable a_struct.a_member.a_nested_member`.

Sadly, the output of `frame variable` can be very limiting and doesn't show
type names by default, nor lets you easily query specific information about
a type or a variable (e.g. is this int type signed or unsigned?).

## Python SB tests

These tests are based on the public LLDB SB API and can be extremely precise.
I recommend looking for the following files as reference for how to query
variable and type information:

- third-party/llvm-project/lldb/include/lldb/API/SBFrame.h
- third-party/llvm-project/lldb/include/lldb/API/SBValue.h
- third-party/llvm-project/lldb/include/lldb/API/SBType.h

You can use the following example as a starter test:

```python
from lib.TestBase import TestBase


class TestSample(LLDBTestBase):
    # The base class sets up the debugger

    def test_sample(self):
        # build_and_launch builds the given file located in the Inputs/
        # directory, then creates a target with the resultant binary, places
        # breakpoints on all the locations with the `# breakpoint` comment, and
        # yields at the first stop.
        with self.build_and_launch("input_file.mojo") as ctx:
            # The ctx variable contains the process, thread, and frame at the
            # stop point, which can be used to query information.
            var = ctx.frame.FindVariable("a_variable")
            assert var.GetValue() == "420"
            # We can also resume the process to the next breakpoint
            next_cxt = ctx.resume()
            assert next_ctx
            assert next_ctx.frame.FindVariable("another_variable").GetValue() == "300"

```

For a more detailed example, you can take a look at `test_sample.py`, which
shows some additional variable and type queries.

For further details, the list of local variables can be accessed through
`StopContext.frame.FindVariable()` or
`StopContext.frame.FindVariableGetValueForVariablePath()`, which resemble the
`frame variable` command. Once you have a variable, which is an `SBValue`
object, you can start querying its value if it's a primitive via the
`GetValue()` family of functions, its summary if it has a synthetic formatter
via `GetSummary()`, or access its fields via `GetChildAtIndex`. Similarly, the
type can be gotten via `GetType()`.

As a final tip, don't hardcode line numbers in your tests. They break often, so
it's better to add markers in the source code and do string search for them
to use them in asserts. The class `SourceFile` should help with that.
