# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s
# RUN: kgen-translate -import-mojo %s --mlir-print-debuginfo --debug-level full -o /dev/null

# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample:
  var x : Int
  fn __init__(inout self): self.x = 42; pass
  fn noop(self): pass
  fn __moveinit__(inout self, owned existing: Self): self.x = existing.x
  fn __copyinit__(inout self, existing: Self): self.x = existing.x
  fn __bool__(self) -> Bool: return True

  # Destructor should not recurse.
  # CHECK-LABEL: lit.func @"__del__
  # CHECK-NEXT:    [[IMMREF:%.*]] = lit.ref.immut %self
  # CHECK-NEXT:    lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT:    %none = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT:    lit.ownership.mark_destroyed %self
  # CHECK-NEXT:    kgen.return %none : !kgen.none
  fn __del__(owned self):
    self.noop()

fn consume(owned a: MemExample): pass

struct MemPair:
  var a: MemExample
  var b: MemExample
  fn __init__(inout self):
    self.a = self.b := MemExample()

  fn use(self): pass


# CHECK-LABEL: lit.struct.decl @RegExample
# CHECK: attributes {{.*}}destructor = #kgen.symbol.constant<@ownership::@RegExample::@"__del__
@register_passable
struct RegExample:
  fn __init__() -> Self:
    return RegExample{}
  fn __copyinit__(self) -> Self: # CHECK: lit.func @"__copyinit__
    return RegExample{}

  fn noop(self): pass
  # CHECK-LABEL: lit.func @"__del__
  # CHECK-NEXT: %self_0 = lit.var.decl "self" imp
  # CHECK-NEXT: lit.ref.store %self, %self_0
  # CHECK-NEXT:  = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT: lit.ownership.mark_destroyed %self_0
  # CHECK-NEXT: kgen.return
  fn __del__(owned self):
    pass

  fn mutate(inout self):
    pass

fn consume(owned a: RegExample): pass

# CHECK-LABEL: lit.func @"destructors
# CHECK-SAME: (%arg0: !lit.ref<!MemExample, mut {{.*}}> owned_in_mem)
fn destructors(owned arg0: MemExample):
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%arg0)

  # CHECK-NEXT: %mem1 = lit.var.decl "mem1" var
  var mem1 = MemExample()
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem1)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem1)

  var mem2 = MemExample()
  # CHECK-NEXT: %mem2 = lit.var.decl "mem2" var
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem2)
  mem2.noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem2
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)

  mem2 = MemExample()
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem2)

  var reg = RegExample()
  # CHECK-NEXT: %reg = lit.var.decl "reg"
  # CHECK-NEXT: [[REGVAL:%.*]] = lit.call {{.*}}__init__
  # CHECK-NEXT: lit.ref.store [[REGVAL]], %reg
  # CHECK-NEXT: [[REG:%.*]] = lit.ref.load %reg
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REG]])

  mem2.noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem2
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)

  # CHECK-NEXT:  %mem3 = lit.var.decl "mem3"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem3)
  let mem3 = MemExample()

  # Test pointless transfers from RValues and trivial values.
  # These should warn and not create IR transfers.

  # First transfer is ok.
  # CHECK-NEXT: [[T1:%.*]] = lit.transfer_mem_ownership %mem3
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[T1]])
  consume(mem3^^^)

  # CHECK-NEXT: %anonymous2A = lit.var.decl "anonymous
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%anonymous2A)
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}(%anonymous2A)
  consume(MemExample()^)

  # CHECK-NEXT: %someInt = lit.var.decl
  # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}4
  # CHECK-NEXT: lit.ref.store [[FOUR]], %someInt
  # CHECK-NEXT: [[FIVE:%.*]] = kgen.param.constant: {{.*}}5
  # CHECK-NEXT: lit.ref.store [[FIVE]], %someInt
  var someInt = 4
  someInt = 5  # silence let warning.
  # CHECK-NEXT: = lit.ref.load %someInt
  _ = someInt^

  # CHECK-NEXT: [[REG:%.*]] = kgen.param.materialize: !RegExample
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[REG]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REG]])
  RegExample{}.noop()

  # CHECK-NEXT: %localReg = lit.var.decl
  # CHECK-NEXT: [[REG2:%.*]] = lit.call {{.*}}@RegExample::@"__init__()"()
  # CHECK-NEXT: lit.ref.store [[REG2]], %localReg
  # CHECK-NEXT: [[REG:%.*]] = lit.ref.load %localReg
  # CHECK-NEXT: [[REG2C:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[REG]])
  # CHECK-NEXT: [[REG:%.*]] = lit.ref.load %localReg
  # CHECK-NEXT: [[BIGREG:%.*]] = lit.struct.create(a=[[REG2C]], b=[[REG]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BIGREG]])
  var localReg = RegExample()
  _ = BigRegExample{a: localReg, b: localReg }

# CHECK-LABEL: lit.func @"indirect_call
fn indirect_call[detail_fn: fn() -> MemExample]():
       # CHECK: %mem = lit.var.decl
       # CHECK-NEXT: lit.call_param{{.*}}(%mem)
       let mem = detail_fn()
       # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem
       # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
       mem.noop()
       # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%mem)

# CHECK-LABEL: lit.struct.decl @Parameterized<level: !Int>
struct Parameterized[level: Int]:
    fn __init__(inout self): pass

    fn __del__(owned self):
        pass

# CHECK-LABEL: lit.func @"test_parameterized
fn test_parameterized():
  # CHECK: %x = lit.var.decl "x"
  let x = Parameterized[4]()
  # CHECK: lit.call {{.*}}@"__init__{{.*}}(%x)
  # CHECK: lit.call {{.*}}__del__{{.*}}<:!Int {4}>(%x)

struct Complicated:
  var a: MemExample
  var b: MemExample

# Issue #12068 - This shouldn't crash.
fn testPointerGEP(ptr: __mlir_type[`!kgen.pointer<`, Complicated, `>`]) -> Int:
  let addr = ptr
  return __get_address_as_lvalue(addr).b.x

# This exercises turning a pop.pointer into an RValue, which produces an 'owned'
# pointer magically from memory.
# CHECK-LABEL: lit.func @"testTakePointeeAsOwned
fn testTakePointeeAsOwned(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`],
                          i1ptr: __mlir_type.`!kgen.pointer<i1>`):
  # This should run the destructor.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REF]])
  _ = __get_address_as_owned_value(ptr)

  # This should run the destructor and not get omitted.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REF]])
  _ = __get_address_as_owned_value(ptr)

  # The RValue can be consumed directly.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[REF]])
  consume(__get_address_as_owned_value(ptr))

  # i1 doesn't have ownership but should still work for generality.
  # CHECK-NEXT: %ownedI1 = lit.var.decl
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %i1ptr end_uninit :
  # CHECK-NEXT: [[I1VAL:%.*]] = lit.load.consume [[REF]]
  # CHECK-NEXT: lit.ref.store [[I1VAL]], %ownedI1
  var ownedI1 = __get_address_as_owned_value(i1ptr)

  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>


# CHECK-LABEL: testGetAsUnitializedObject
fn testGetAsUnitializedObject(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`],
                             i1ptr: __mlir_type.`!kgen.pointer<i1>`):
  # Overwriting the value in a __get_address_as_lvalue forces it to be destroyed
  # before the overwrite.

  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr :
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])
  __get_address_as_lvalue(ptr) = MemExample()

  # Overwriting the value in a __get_address_as_uninit_lvalue does not destroy
  # the memory, because it is uninit.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr start_uninit :
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])
  __get_address_as_uninit_lvalue(ptr) = MemExample()

  # i1 doesn't have ownership but should still work for generality.
  # CHECK-NEXT: %i1Val = lit.var.decl "i1Val"
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %i1ptr :
  # CHECK-NEXT: [[I1VAL:%.*]] = lit.ref.load [[REF]]
  # CHECK-NEXT: lit.ref.store [[I1VAL]], %i1Val
  var i1Val = __get_address_as_lvalue(i1ptr)

  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: testCondGetAsUnitializedObject
# Early exit from fn using __get_address_as_uninit_lvalue should work.
# https://github.com/modularml/modular/issues/27472
fn testCondGetAsUnitializedObject(exit_early: __mlir_type.i1,
                                  ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`]):
  # CHECK: hlcf.if
  if exit_early:
      return

  # CHECK: [[REF:%.*]] = lit.ref.from_pointer %ptr start_uninit :
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])
  # CHECK-NEXT: %none = kgen.param.constant: none
  # CHECK-NEXT: kgen.return %none
  __get_address_as_uninit_lvalue(ptr) = MemExample()


# CHECK-LABEL: lit.struct.decl @FieldSensitiveMemExample
struct FieldSensitiveMemExample:
  var f1 : MemExample
  var f2 : MemExample

  # CHECK: lit.func @"__init__
  fn __init__(inout self):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%0)
    self.f1 = MemExample()
    # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f2]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%2)
    self.f2 = MemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return

  # CHECK: lit.func @"__init__
  fn __init__(inout self, a: MemExample, b: MemExample):
    self.f1 = a
    self.f2 = b

  fn __copyinit__(inout self, existing: Self):
    self = Self(existing.f1, existing.f2)

  # CHECK-LABEL: lit.func @"mutate
  fn mutate(inout self):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%0)

    # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%2)
    self.f1 = MemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>

 # CHECK-LABEL: lit.func @"mutate2
  fn mutate2(inout self):
    # Disable the dtor of 'f1' before we overwrite it to show we can do this.
    # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.ownership.mark_destroyed [[F1R]]
    __mlir_op.`lit.ownership.mark_destroyed`[_type=None](
       __get_ref_from_value(self.f1))

    # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[F1R]])
    self.f1 = MemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>


  # CHECK-LABEL: lit.func @"__del__
  # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%0)
  # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f2]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%2)
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: lit.ownership.mark_destroyed %self

# This disables the destructor of 'x' which causes the fields to be destroyed.
# CHECK-LABEL: lit.func @"disableDtor
fn disableDtor(owned x: FieldSensitiveMemExample):
  # CHECK-NEXT: %0 = lit.ref.struct.ger %x[f1]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%0)
  # CHECK-NEXT: %2 = lit.ref.struct.ger %x[f2]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%2)
  # CHECK-NEXT: lit.ownership.mark_destroyed %x
  # CHECK-NEXT: kgen.param.constant: none
  __mlir_op.`lit.ownership.mark_destroyed`[_type=None](
       __get_ref_from_value(x))

# CHECK-LABEL: lit.func @"regpassable_owned_args_mutable
fn regpassable_owned_args_mutable(owned x: RegExample):
  # CHECK-NEXT: %x_0 = lit.var.decl "x" imp
  # CHECK-NEXT: lit.ref.store %x, %x_0
  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x_0)
  x.mutate()

  # CHECK-NEXT: [[X:%.*]] = lit.ref.load %x_0
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[X]])
  # CHECK-NEXT: [[X:%.*]] = {{.*}}"__init__
  # CHECK-NEXT: lit.ref.store [[X]], %x_0
  x = RegExample()

  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x_0)
  x.mutate()
  # CHECK-NEXT: [[X:%.*]] = lit.ref.load %x_0
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[X]])

# Result optimization cannot emit directly into a value that is passed as an
# argument, because this forms a mutable reference to something immutably
# borrowed implicitly.  We must invoke the copy ctor.
# CHECK-LABEL: lit.func @"use_and_return
fn use_and_return(a: FieldSensitiveMemExample) -> FieldSensitiveMemExample:
  # This will read from 'a' and write into the result slot in an arbitrary
  # order. They cannot alias.
  return FieldSensitiveMemExample(a.f2, a.f1)

fn use_and_return2(a: FieldSensitiveMemExample) -> MemExample:
  return a.f2

# CHECK-LABEL: lit.func @"test_result_optimization
fn test_result_optimization():
  # CHECK-NEXT: %example = lit.var.decl "example"
  # CHECK-NEXT: lit.call @{{.*}}"__init__{{.*}}(%example)
  var example = FieldSensitiveMemExample()

  # Direct reuse of the result slot forces a temporary.

  # CHECK: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
  # CHECK-NEXT: lit.call @ownership{{.*}}(%__call_result_tmp__, [[IMMREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%example)
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %__call_result_tmp__
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}(%example, [[IMMREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%__call_result_tmp__)
  example = use_and_return(example)

  # Aliased reuse of part of the result slot forces a temporary.

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: [[F1:%.*]] = lit.ref.struct.ger %example[f1]
  # CHECK-NEXT: %__call_result_tmp___0 = lit.var.decl
  # CHECK-NEXT: lit.call @ownership::@"use_and_return2{{.*}}(%__call_result_tmp___0, [[IMMREF]])
  example.f1 = use_and_return2(example)
  # CHECK-NEXT: [[F1_2:%.*]] = lit.ref.struct.ger [[IMMREF]][f1]
  # CHECK-NEXT: [[MUTREF:%.*]] = kgen.rebind [[F1_2]]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[MUTREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"__moveinit__{{.*}}([[F1]], %__call_result_tmp___0)

  example.mutate()
  # CHECK-NEXT: lit.call @{{.*}}@"mutate{{.*}}(%example)
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%example)

  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>

##===----------------------------------------------------------------------===##
# Consume Expressions
##===----------------------------------------------------------------------===##

# CHECK: lit.func @"test_result_consume_reg
fn test_result_consume_reg(cond: __mlir_type.i1) -> RegExample:
  # CHECK-NEXT: %example2 = lit.var.decl
  # CHECK: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: lit.ref.store [[TMP]], %example2
  var example2 = RegExample()

  # CHECK-NEXT: hlcf.if %cond
  if (cond):
    # CHECK-NEXT: [[TMP:%.*]] = lit.transfer_mem_ownership %example2
    # CHECK-NEXT: [[TMP2:%.*]] = lit.load.consume [[TMP]]
    # CHECK-NEXT: kgen.return [[TMP2]]
    return example2^
  else: # CHECK-NEXT: } else {
    # CHECK-NEXT: [[TMP2:%.*]] = lit.ref.load %example2
    # CHECK-NEXT: kgen.return [[TMP2]]
    return example2  # copy/del -> move optimization.

# CHECK: lit.func @"consumeMem
fn consumeMem(owned x: MemExample):
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%x)
  # CHECK-NEXT: kgen.param.constant: none
  pass

# CHECK: lit.func @"test_result_consume_mem
fn test_result_consume_mem(cond: __mlir_type.i1) -> MemExample:
  # CHECK-NEXT: %example = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%example)
  let example = MemExample()

  # This doesn't consume example, so it must copy it. It does consume the copy.
  # CHECK-NEXT: %anonymous2A = lit.var.decl
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, [[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}consumeMem{{.*}}(%anonymous2A)
  consumeMem(example)

  # This does consume example, so no copy needed.
  # CHECK-NEXT: [[CONSUME:%.*]] = lit.transfer_mem_ownership %example
  # CHECK-NEXT: lit.call {{.*}}consumeMem{{.*}}([[CONSUME]])
  consumeMem(example^)

  # CHECK-NEXT: %example2 = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%example2)
  let example2 = MemExample()

  # CHECK-NEXT: [[CONSUME:%.*]] = lit.transfer_mem_ownership %example2
  # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%__result__, [[CONSUME]])
  # CHECK-NEXT: kgen.param.constant: none
  return example2^

# CHECK-LABEL: lit.struct.decl @BigRegExample
@register_passable
struct BigRegExample:
  var a: RegExample
  var b: RegExample

  # CHECK-LABEL: lit.func @"__init__()"
  fn __init__() -> Self:
    # CHECK-NEXT: %0 = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: %1 = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: %2 = lit.struct.create(a=%0, b=%1)
    # CHECK-NEXT: kgen.return %2
    return BigRegExample{a: RegExample(), b: RegExample() }

  # CHECK-LABEL: lit.func @"__copyinit__
  fn __copyinit__(self) -> Self:
    # CHECK-NEXT: %0 = lit.struct.extract %self[a]
    # CHECK-NEXT: %1 = lit.struct.extract %self[b]
    # CHECK-NEXT: %2 = lit.call {{.*}}__copyinit__{{.*}}(%0)
    # CHECK-NEXT: %3 = lit.call {{.*}}__copyinit__{{.*}}(%1)
    # CHECK-NEXT: %4 = lit.struct.create(a=%2, b=%3)
    # CHECK-NEXT: kgen.return %4
    return BigRegExample{a: self.a, b: self.b }

  # CHECK-LABEL: lit.func @"__del__
  # CHECK-NEXT: %self_0 = lit.var.decl "self" imp
  # CHECK-NEXT: lit.ref.store %self, %self_0

  # CHECK-NEXT: [[APTR:%.*]] = lit.ref.struct.ger %self_0[a]
  # CHECK-NEXT: [[AVAL:%.*]] = lit.ref.load [[APTR]]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[AVAL]])
  # CHECK-NEXT: [[BPTR:%.*]] = lit.ref.struct.ger %self_0[b]
  # CHECK-NEXT: [[BVAL:%.*]] = lit.ref.load [[BPTR]]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BVAL]])
  # CHECK-NEXT:  = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT: lit.ownership.mark_destroyed %self_0
  # CHECK-NEXT: kgen.return


# CHECK-LABEL: lit.func @"bigreg_test
fn bigreg_test():
  # CHECK-NEXT: %varThing = lit.var.decl "varThing"
  # CHECK-NEXT: [[INITVAL:%.*]] = lit.call {{.*}}__init__()
  # CHECK-NEXT: lit.ref.store [[INITVAL]], %varThing
  var varThing = BigRegExample()

  # CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: [[LIFEEND:%.*]] = lit.transfer_mem_ownership [[FIELD]]
  # CHECK-NEXT: [[AVAL:%.*]] = lit.load.consume [[LIFEEND]]
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[AVAL]])
  consume(varThing.a^)

  # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %varThing[b]
  # CHECK-NEXT: [[BVAL:%.*]] = lit.ref.load [[BREF]]
  # CHECK-NEXT: [[BCOPY:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[BVAL]])
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[BCOPY]])
  consume(varThing.b)

  # CHECK-NEXT: [[NEW:%.*]] = lit.call {{.*}}__init__
  # CHECK-NEXT: [[AREF:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: lit.ref.store [[NEW]], [[AREF]]
  # CHECK-NEXT: [[OLDVAL:%.*]] = lit.ref.load %varThing
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[OLDVAL]])
  varThing.a = RegExample()

  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: lit.struct.decl @ExoticDelExample
@register_passable
struct ExoticDelExample:
  var cond: __mlir_type.i1
  var b: BigRegExample
  var c: RegExample

 # CHECK-LABEL: lit.func @"__del__
  fn __del__(owned self):
    # CHECK-NEXT: %self_0 = lit.var.decl "self" imp
    # CHECK-NEXT: lit.ref.store %self, %self_0

    # self.b gets destroyed ASAP since it isn't used.
    # CHECK-NEXT: [[BPTR:%.*]] = lit.ref.struct.ger %self_0[b]
    # CHECK-NEXT: [[BVAL:%.*]] = lit.ref.load [[BPTR]]
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BVAL]])

    # Test the condition
    # CHECK-NEXT: [[CONDPTR:%.*]] = lit.ref.struct.ger %self_0[cond]
    # CHECK-NEXT: [[CONDVAL:%.*]] = lit.ref.load [[CONDPTR]]
    # CHECK-NEXT: hlcf.if [[CONDVAL]] {
    if self.cond:
      # This side we manually consume for c.

      # CHECK-NEXT: [[CREF:%.*]] = lit.ref.struct.ger %self_0[c]
      # CHECK-NEXT: [[CREF2:%.*]] = lit.transfer_mem_ownership [[CREF]]
      # CHECK-NEXT: [[CVAL:%.*]] = lit.load.consume [[CREF2]]
      # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[CVAL]])
      # CHECK-NEXT: hlcf.yield
      consume(self.c^)

    # CHECK-NEXT: } else {
    # Destroy C automatically on the else side.
    # CHECK-NEXT:  [[CPTR:%.*]] = lit.ref.struct.ger %self_0[c]
    # CHECK-NEXT:  [[CVAL:%.*]] = lit.ref.load [[CPTR]]
    # CHECK-NEXT:  lit.call {{.*}}__del__{{.*}}([[CVAL]])
    # CHECK-NEXT:  hlcf.yield
    # CHECK-NEXT:}

    # CHECK-NEXT: = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT:lit.ownership.mark_destroyed %self_0
    # CHECK-NEXT:kgen.return


# CHECK-LABEL: lit.func @"def_borrowed
# CHECK-SAME: %a: !lit.ref<!MemExample, imm {{.*}}> borrow_in_mem
def def_borrowed(borrowed a: MemExample) -> None:
  # CHECK-NEXT: kgen.param.constant: none
  pass


# https://github.com/modularml/modular/issues/24161
@register_passable("trivial")
struct AddrSpace:
    var _value: __mlir_type.index
    fn __init__(value: __mlir_type.index) -> Self:
        return Self{_value:value}
    fn value(self) -> __mlir_type.index:
        return self._value
@value
@register_passable("trivial")
struct MemExamplePtr[addrspace: AddrSpace = __mlir_attr.`0:index`]:
    var value: __mlir_type[
        `!kgen.pointer<`, MemExample, `, `, addrspace.value(), `>`
    ]

fn sadge(ptr: MemExamplePtr[]):
    __get_address_as_uninit_lvalue(ptr.value) = MemExample()
    return


trait SomeTrait:
    pass

struct GenericType(SomeTrait):
    fn __del__(owned self):
        pass

@register_passable
struct GenericRegType(SomeTrait):
    fn __del__(owned self):
        pass

# CHECK-LABEL: lit.func @"destruct_generic_return
fn destruct_generic_return():
    @parameter
    fn return_generic_type[T: SomeTrait]() -> T:
        while True:
            pass

    # CHECK: call {{.*}}@GenericType::@"__del__
    _ = return_generic_type[GenericType]()
    # CHECK: call {{.*}}@GenericRegType::@"__del__
    _ = return_generic_type[GenericRegType]()


# CHECK-LABEL: lit.struct.decl @RegisterExistingDtor
@register_passable
struct RegisterExistingDtor:
    # CHECK: lit.func @"__del__{{.*}}_thunk"
    fn __del__(owned self):
        # CHECK: call {{.*}}@"__del__
        pass

@register_passable
struct RegisterNoDtor:
    pass

struct MemoryNoDtor:
    pass


# CHECK-LABEL: lit.struct.decl @RegExampleValue({{.*}}) register_passable
# Compiler crashes trying to insert a destructor call
# https://github.com/modularml/modular/issues/26410
@value
@register_passable
struct RegExampleValue:
  var x: RegExample
  fn __init__() -> Self:
    return Self {x: RegExample()}

  # Make sure the synthesized dtor is taken register style.
  # CHECK: lit.func @"__del__{{.*}}(%self: !RegExampleValue, |)
  # CHECK-NEXT: %self_0 = lit.var.decl "self"
  # CHECK-NEXT: lit.ref.store %self, %self_0
  # CHECK: kgen.param.constant: none
  # CHECK: lit.ownership.mark_destroyed %self_0

# [Bug] __result__ is uninitialized
# https://github.com/modularml/modular/issues/27792
# CHECK-LABEL: lit.func @"test_or
fn test_or(a: MemExample) -> MemExample:
  # CHECK: hlcf.if {{.*}} {
  # CHECK:   lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %a)
  # CHECK: } else {
  # CHECK:   lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %a)
  # CHECK: }
  return a or a


# ===----------------------------------------------------------------------=== #
# Variadics
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.func @"variadic_mems
# CHECK-SAME: [imm *"mems`"](
# CHECK-SAME: %mems: !kgen.variadic<!lit.ref<!MemExample, imm *"mems`">, borrow_in_mem> borrow)
fn variadic_mems(*mems: MemExample):
  # CHECK-NEXT: %mems_0 = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}@VariadicListMem::@"__init__
  # CHECK-SAME: <:!AnyType #MemExample{{.*}}:lifetime<0> *"mems`">(%mems_0, %mems)
  pass

# CHECK-LABEL: lit.func @"call_variadic_mems
fn call_variadic_mems(a: MemExample, b: MemExample):
  # CHECK-NEXT: %0 = kgen.rebind %a : !lit.ref<!MemExample, imm *"a`"> to !lit.ref<!MemExample, imm {*"a`", *"b`"}>
  # CHECK-NEXT: %1 = kgen.rebind %b : !lit.ref<!MemExample, imm *"b`"> to !lit.ref<!MemExample, imm {*"a`", *"b`"}>
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.create [%0, %1]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[imm {*"a`", *"b`"}]([[VAR]])
  variadic_mems(a, b)

  # Variadic use keeps the memory value alive.
  # CHECK-NEXT: %c = lit.var.decl "c"
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%c, %a)
  let c = a
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.splat 1, [[IMMREF]]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[muttoimm *"c`0"]([[VAR]])
  variadic_mems(c)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%c)
  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: lit.func @"variadic_field_sensitivity
fn variadic_field_sensitivity():
  # Test that we field sensitively track variadics.
  # CHECK:  %memPair = lit.var.decl
  var memPair = MemPair()

  # CHECK: [[AREF:%.*]] = lit.ref.struct.ger %memPair[a]
  # CHECK-NEXT: [[OWNEDAREF:%.*]] = lit.transfer_mem_ownership [[AREF]]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[OWNEDAREF]])
  _ = memPair.a^  # Destroy a.

  # Can still pass b through varargs.
  # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %memPair[b]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[BREF]]
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.splat 1, [[IMMREF]]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[muttoimm *"memPair`0"]([[VAR]])
  variadic_mems(memPair.b)

  # Need to restore 'a' so memPair may destruct.
  # CHECK: [[AREF:%.*]] = lit.ref.struct.ger %memPair[a]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[AREF]])
  memPair.a = MemExample()

  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%memPair)
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.func @"variadic_inout_mems
# CHECK-SAME: [mut *"mems`"](
# CHECK-SAME: %mems: !kgen.variadic<!lit.ref<!MemExample, mut *"mems`">, byref> borrow)
fn variadic_inout_mems(inout *mems: MemExample):
  # CHECK-NEXT: %mems_0 = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}@VariadicListMem::@"__init__
  # CHECK-SAME: <:!AnyType #MemExample{{.*}} :lifetime<1> *"mems`">(%mems_0, %mems)
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mems_0 :
  # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__refitem__{{.*}}([[IMMREF]], [[ZERO]])
  # CHECK-NEXT: [[MEMREF:%.*]] = lit.call {{.*}}__mlir_ref__{{.*}}([[REF]])
  # CHECK-NEXT: [[XREF:%.*]] = lit.ref.struct.ger [[MEMREF]][x]
  # CHECK-NEXT: [[ONE:%.*]] = kgen.param.constant
  # CHECK-NEXT: lit.call {{.*}}__iadd__{{.*}}([[XREF]], [[ONE]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mems_0)
  mems[0].x += 1

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.func @"call_variadic_inout_mems
fn call_variadic_inout_mems():
  var a = MemExample()
  var b = MemExample()
  # CHECK: [[AR:%.*]] = kgen.rebind %a : !lit.ref<!MemExample, mut *"a`0"> to !lit.ref<!MemExample, mut {*"a`0", *"b`1"}>
  # CHECK-NEXT: [[BR:%.*]] = kgen.rebind %b : !lit.ref<!MemExample, mut *"b`1"> to !lit.ref<!MemExample, mut {*"a`0", *"b`1"}>
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.create [[[AR]], [[BR]]]
  # CHECK-NEXT: lit.call {{.*}}variadic_inout_mems{{.*}}[mut {*"a`0", *"b`1"}]([[VAR]])
  variadic_inout_mems(a, b)

  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%b)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.func @"variadic_inout_mems_iter
fn variadic_inout_mems_iter(inout *mems: MemExample):
  # Verify the iterator keeps the VariadicListMem alive.
  # CHECK-NEXT: %mems_0 = lit.var.decl

  # CHECK: %iter = lit.var.decl
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mems_0 :
  # CHECK-NEXT: lit.call {{.*}}__iter__{{.*}}(%iter, [[IMMREF]])
  var iter = mems.__iter__()

  # CHECK-NEXT: %x = lit.var.decl
  # CHECK-NEXT: [[ELTREF:%.*]] = lit.call {{.*}}__next__{{.*}}(%iter)

  ## FIXME: This destruction should be ordered after the destroy of the iterator
  ## Since the iterator can refer to the mems struct.
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mems_0)

  # Iterator is destroyed as soon as we're done with it.
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%iter)

  # __next__ returns a Reference which needs to turn in to !lit.ref
  # CHECK-NEXT: [[ELTDEREF:%.*]] = lit.call {{.*}}__refitem__{{.*}}([[ELTREF]])
  # CHECK-NEXT: [[ELTDEREFIMM:%.*]] = lit.ref.immut [[ELTDEREF]]
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%x, [[ELTDEREFIMM]])
  let x : MemExample = iter.__next__()[]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%x)

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return



# CHECK-LABEL: lit.func @"test_partial_overwrite
fn test_partial_overwrite(cond: __mlir_type.i1):
  # CHECK-NEXT: %pair = lit.var.decl "pair"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%pair)
  var pair = MemPair()

  # CHECK-NEXT: hlcf.if %cond {
  if cond:
    # Inserted destruction of incoming pair.b
    # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %pair[b]
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BREF]])

    # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %pair[b]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[BREF]])
    pair.b = MemExample()

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %pair
    # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[IMMREF]])
    pair.use()
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%pair)
    # CHECK-NEXT: hlcf.yield
  else: # CHECK-NEXT: } else {
    # Inserted destruction of whole pair.
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%pair)

    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return
    return
  # CHECK-NEXT: }
