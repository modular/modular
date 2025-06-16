# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Callable

import max._core

# Many of the generated overloads for constructors are more specialized in
# C++ than they are in Python. For example, `int32_t` and `int64_t` and `size_t`
# all map to `int` in Python typing. It may not always be clear which of these
# overloads will be run for a given set of inputs (though in most cases it's the first one)
# but we disable mypy errors for shadowed overloads.
#
# mypy: disable-error-code="overload-cannot-match"

# DiagnosticHandlers aren't a thing that Python can reasonably provided. In most cases
# these are automatically provided, but there are a few custom verifiers not covered yet.
# This binding prevents errors in those cases.
DiagnosticHandler = Callable

# This is a bug I haven't yet chased down in Nanobind's type renderings.
# - In some circumstances, `max._core.dialicts.mosh.ShapeType` is being shortened
#   to `h.ShapeType`, which obviously doesn't exist.
# - I haven't figured out a clean repro or workaround, I suspect it's some awkward case
#   with `nb_func_render_signature` because for instance adding garbage characters to the
#   `const_name` in the type caster will cause it to repro in different places.
# - For now, really hacky thing to work around.

def LegalizeRMOOps(skip_rmo_rebind: bool = False) -> max._core.Pass:
    """
    There are broadly two categories of RMO operators. The first are those
    analogous to an existing MO operator. These are lowered by properly
    rebinding each input and result to the types determined in the `MOAnalogue`
    interface. These operators have a lowering automatically generated as long
    as the `MOAnalogue` operator is implemented.

    The next are RMO operators which lower to other RMO operators.
    """
