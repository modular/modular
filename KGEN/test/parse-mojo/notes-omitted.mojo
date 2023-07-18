# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that the default value for `--max-notes-per-diagnostic` is 10.
# RUN: not %mojo %s 2>&1 | FileCheck %s

# Test that the option controls this setting.
# RUN: not %mojo --max-notes-per-diagnostic 5 %s \
# RUN:   2>&1 | FileCheck %s --check-prefix CHECK-FIVE

# fmt: off
struct s1: pass
struct s2: pass
struct s3: pass
struct s4: pass
struct s5: pass
struct s6: pass
struct s7: pass
struct s8: pass
struct s9: pass
struct s10: pass
struct s11: pass

fn go10(x: s1): pass
fn go10(x: s2): pass
fn go10(x: s3): pass
fn go10(x: s4): pass
fn go10(x: s5): pass
fn go10(x: s6): pass
fn go10(x: s7): pass
fn go10(x: s8): pass
fn go10(x: s9): pass
fn go10(x: s10): pass

fn go11(x: s1): pass
fn go11(x: s2): pass
fn go11(x: s3): pass
fn go11(x: s4): pass
fn go11(x: s5): pass
fn go11(x: s6): pass
fn go11(x: s7): pass
fn go11(x: s8): pass
fn go11(x: s9): pass
fn go11(x: s10): pass
fn go11(x: s11): pass

fn main():
  # CHECK-NOT: {{.*}} (0 more notes omitted.)
  # CHECK-FIVE: {{.*}} (5 more notes omitted.)
  go10(0)
  # CHECK: {{.*}} (1 more notes omitted.)
  # CHECK-FIVE: {{.*}} (6 more notes omitted.)
  go11(0)
