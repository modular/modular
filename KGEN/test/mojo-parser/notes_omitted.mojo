# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated --max-notes-per-diagnostic=2 --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s

# fmt: off
struct s1(Movable where False): pass
struct s2(Movable where False): pass
struct s3(Movable where False): pass
struct s4(Movable where False): pass
struct s5(Movable where False): pass
struct s6(Movable where False): pass
struct s7(Movable where False): pass
struct s8(Movable where False): pass
struct s9(Movable where False): pass
struct s10(Movable where False): pass
struct s11(Movable where False): pass

def go10(x: s1): pass
def go10(x: s2): pass
def go10(x: s3): pass
def go10(x: s4): pass
def go10(x: s5): pass
def go10(x: s6): pass
def go10(x: s7): pass
def go10(x: s8): pass
def go10(x: s9): pass
def go10(x: s10): pass

def go11(x: s1): pass
def go11(x: s2): pass
def go11(x: s3): pass
def go11(x: s4): pass
def go11(x: s5): pass
def go11(x: s6): pass
def go11(x: s7): pass
def go11(x: s8): pass
def go11(x: s9): pass
def go11(x: s10): pass
def go11(x: s11): pass

def foo():
  # CHECK: 8 more notes omitted
  go10(__mlir_attr.`0 : index`)
  # CHECK: 9 more notes omitted
  go11(__mlir_attr.`0 : index`)
