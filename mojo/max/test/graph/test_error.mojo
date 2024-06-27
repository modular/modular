# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s

import testing

from max.graph import Graph, Type
from max.graph.error import error, format_system_stack


def test_error():
    g = Graph(List[Type]())
    with testing.assert_raises(contains="test_error.mojo:17"):
        raise error(g, "blah")
    with testing.assert_raises(contains="foo.bar - blah"):
        with g.layer("foo"):
            with g.layer("bar"):
                raise error(g, "blah")


def test_format_system_stack():
    stack_trace = format_system_stack()
    testing.assert_true("System stack:" in stack_trace)


def main():
    test_error()
    test_format_system_stack()
