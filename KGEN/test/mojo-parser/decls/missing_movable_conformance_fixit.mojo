# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated %s --mojo-diagnose-missing-movable-conformance --experimental-export-fixit=%t.yaml -o /dev/null 2>&1 | FileCheck %s
# RUN: FileCheck %s --check-prefix=YAML < %t.yaml

# Regression test for two review findings on the --experimental-export-fixit
# path in kgen-translate:
#
# 1. Fix-its must be collected under the tool's DEFAULT configuration,
#    without also having to pass -use-mlir-diagnostics=false. Fix-it
#    collection only happens on the SourceMgr diagnostic path, so this
#    RUN line deliberately omits that flag to prove it's forced on
#    automatically whenever a fix-it flag is requested.
# 2. A fix-it collected on an earlier, successfully-resolved struct must
#    still be exported even though a later fatal parse error aborts the
#    overall translation -- fix-it handling must not be skipped just
#    because the parse ultimately fails.

# CHECK: warning: struct does not explicitly conform to 'Movable'
struct NoConformanceList:
    pass


# YAML: ReplacementText:

# Regression test: a trailing `where` clause with no existing conformance
# parens used to get the synthesized `(...)` list glued onto `where` with no
# separating space. The exported replacement text must retain a trailing
# space so the fixed-up source reads `... (Movable where False) where ...`.
# CHECK: warning: struct does not explicitly conform to 'Movable'
struct PredicateOnStructOuter[value: Int] where value >= 0:
    pass


# YAML: ReplacementText: '(Movable where False) '

# A deliberate syntax error: aborts parsing after the structs above have
# already been resolved (and their fix-its collected).
fn broken(
