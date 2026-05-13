# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --experimental-export-fixit creates a YAML file even when there
# are no fix-its (unlike clang-tidy semantics: we always create the file).
# RUN: rm -f %t3.yaml
# RUN: mojo package --experimental-export-fixit=%t3.yaml %S/test_package_fixit_patch_empty -o %t3.mojoc 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file=%t3.yaml --check-prefix=YAML

# CHECK: Fix-its exported to:
# CHECK: Apply with: 'clang-apply-replacements

# Verify YAML file is created with empty replacements
# YAML: ---
# YAML: MainSourceFile:
# YAML: Replacements:  []
# YAML: ...
