# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestStackTrace(LLDBTestBase):
    def test_stack_trace_format(self):
        """Simple test that ensures frames can be printed out in a nice format.
        It's covering simple parameter types, methods and nested functions, as
        well as printing the values of arguments.

        FIXME(25047): A current limitation when formatting frames is that the
        source name of nested functions lose track of the parameters of the
        parent, because of which `[...]` is printed instead, signaling that some
        parameters are expected, but aren't available.

        TODO(25048): Include argument types in functions.
        """

        with self.build_and_launch("stack-trace.mojo") as ctx:
            frame_descs = [str(frame) for frame in ctx.thread.frames]
            self.assertRegex(
                frame_descs[0],
                (
                    r"stack-trace.Foo\[...\].getParametrized\[...\]"
                    r".nested_function\(z=\(\[0\] = 105.25\)\) at"
                    r" stack-trace.mojo:16:13"
                ),
            )
            self.assertRegex(
                frame_descs[1],
                (
                    r"stack-trace.Foo\[index, index\]"
                    r".getParametrized\[scalar<f32>\]\(self=0x.*,"
                    r" val=\(\[0] = 105.25\)\) at stack-trace.mojo:18:31"
                ),
            )
            self.assertRegex(
                frame_descs[2],
                (
                    r"stack-trace.Foo\[index, index\]"
                    r".getFloat\(self=0x.*, x=\(\[0\] = 1.125\), y=100\)"
                    r" at stack-trace.mojo:21:45"
                ),
            )
            self.assertIn(
                "stack-trace.main() at stack-trace.mojo:25:35", frame_descs[3]
            )
