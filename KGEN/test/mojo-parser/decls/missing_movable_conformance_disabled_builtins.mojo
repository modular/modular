# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mojo-disable-builtins -verify-diagnostics -mojo-diagnose-missing-movable-conformance
# RUN: %parse-mojo-isolated %s -mojo-disable-builtins -mojo-diagnose-missing-movable-conformance --experimental-export-fixit=%t.yaml -o /dev/null
# RUN: FileCheck %s --check-prefix=YAML < %t.yaml

# Regression test: with builtins disabled, `Movable` itself isn't a
# resolvable declaration, so the diagnostic must not fire and must not offer
# a fix-it that would insert an unresolvable reference to it. No
# `expected-warning` annotation below means `-verify-diagnostics` fails if the
# diagnostic fires anyway.
struct NoConformanceListNoBuiltins:
    pass

# YAML: Replacements:{{ *}}[]
