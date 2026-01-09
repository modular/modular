# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --experimental-export-fixit creates a YAML file even when there
# are no fix-its (unlike clang-tidy semantics: we always create the file).
# RUN: rm -f %t.yaml
# RUN: %mojo-build --experimental-export-fixit=%t.yaml %s 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file=%t.yaml --check-prefix=YAML

# CHECK: Fix-its exported to:
# CHECK: Apply with: 'clang-apply-replacements

# Verify YAML file is created with empty replacements
# YAML: ---
# YAML: MainSourceFile:
# YAML: Replacements:  []
# YAML: ...


fn no_fixits_needed():
    pass


def main():
    no_fixits_needed()
