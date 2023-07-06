# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo %s 2>&1 | FileCheck %s

# Testing default --max-notes-per-diagnostic=10

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
  go10(0)
  # CHECK: {{.*}} (1 more notes omitted.)
  go11(0)
