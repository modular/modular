# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample:
  var x : Int
  fn __init__(out self): self.x = 42; pass
  fn noop(self): pass
  fn __moveinit__(out self, owned existing: Self): self.x = existing.x
  fn __copyinit__(out self, existing: Self): self.x = existing.x
  fn __bool__(self) -> Bool: return True

  # Destructor should not recurse.
  # CHECK-LABEL: lit.fn @"__del__
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
  fn __init__(out self):
    self.a = self.b := MemExample()

  fn use(self): pass


# CHECK-LABEL: lit.struct.decl @RegExample
# CHECK: destructor {{.*}}@RegExample::@"__del__
@register_passable
struct RegExample:
  fn __init__(out self):
    return

  @implicit
  fn __init__(out self, value: Int):
    pass

  fn __copyinit__(out self, existing: Self): # CHECK: lit.fn @"__copyinit__
    return

  fn noop(self): pass
  # CHECK-LABEL: lit.fn @"__del__
  # CHECK-NEXT:  = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT: lit.ownership.mark_destroyed %self
  # CHECK-NEXT: kgen.return
  fn __del__(owned self):
    pass

  fn mutate(mut self):
    pass

fn consume(owned a: RegExample): pass

# CHECK-LABEL: lit.fn @"destructors
# CHECK-SAME: (%arg0: !lit.ref<!MemExample, mut {{.*}}> owned_in_mem)
fn destructors(owned arg0: MemExample):
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%arg0)

  # CHECK-NEXT: %mem1 = lit.var.decl "mem1" var
  # expected-warning @+1 {{assignment to 'mem1' was never used}}
  var mem1 = MemExample()
  # CHECK-NEXT: lifetime.start %mem1
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem1)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem1)
  # CHECK-NEXT: lifetime.end %mem1

  var mem2 = MemExample()
  # CHECK-NEXT: %mem2 = lit.var.decl "mem2" var
  # CHECK-NEXT: lifetime.start %mem2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem2)
  mem2.noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem2
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)
  # CHECK-NEXT: lifetime.end %mem2

  mem2 = MemExample()
  # CHECK-NEXT: lifetime.start %mem2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem2)

  # expected-warning @+1 {{assignment to 'reg' was never used}}
  var reg = RegExample()
  # CHECK-NEXT: %reg = lit.var.decl "reg"
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: lifetime.start %reg
  # CHECK-NEXT: lit.ref.store [[TMP]], %reg
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%reg)
  # CHECK-NEXT: lifetime.end

  mem2.noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem2
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)
  # CHECK-NEXT: lifetime.end %mem2

  # CHECK-NEXT: %mem3 = lit.var.decl "mem3"
  # CHECK-NEXT: lifetime.start %mem3
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem3)
  var mem3 = MemExample()

  # Test pointless transfers from RValues and trivial values.
  # These should warn and not create IR transfers.

  # First transfer is ok.
  # CHECK-NEXT: lit.ownership.use %mem3
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}(%mem3)
  # CHECK-NEXT: lifetime.end %mem3
  consume(mem3^^^)

  # CHECK-NEXT: %anonymous2A = lit.var.decl "anonymous
  # CHECK-NEXT: lifetime.start %anonymous2A
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%anonymous2A)
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}(%anonymous2A)
  # CHECK-NEXT: lifetime.end %anonymous2A
  consume(MemExample()^)

  # CHECK-NEXT: %someInt = lit.var.decl
  # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}4
  # CHECK-NEXT: lifetime.start %someInt
  # CHECK-NEXT: lit.ref.store [[FOUR]], %someInt
  # CHECK-NEXT: lifetime.end %someInt
  # CHECK-NEXT: [[FIVE:%.*]] = kgen.param.constant: {{.*}}5
  # CHECK-NEXT: lifetime.start %someInt
  # CHECK-NEXT: lit.ref.store [[FIVE]], %someInt
  # expected-warning @+1 {{assignment to 'someInt' was never used}}
  var someInt = 4
  someInt = 5  # silence var warning.
  # CHECK-NEXT: = lit.ref.load %someInt
  # CHECK-NEXT: lifetime.end %someInt
  _ = someInt^

  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@RegExample::@"__init__{{.*}}()
  # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lit.var.lifetime.start [[ANON]]
  # CHECK-NEXT: lit.ref.store [[TMP]], [[ANON]]
  # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[ANON]]
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMM]])
  RegExample().noop()
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[ANON]])
  # CHECK-NEXT: lit.var.lifetime.end [[ANON]]

  # CHECK-NEXT: %localReg = lit.var.decl
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@RegExample::@"__init__{{.*}}()
  # CHECK-NEXT: lifetime.start %localReg
  # CHECK-NEXT: lit.ref.store [[TMP]], %localReg
  # expected-warning @+1 {{assignment to 'localReg' was never used}}
  var localReg = RegExample()


# CHECK-LABEL: lit.fn @"indirect_call
fn indirect_call[detail_fn: fn() -> MemExample]():
       # CHECK: %mem = lit.var.decl
       # CHECK-NEXT: lifetime.start %mem
       # CHECK-NEXT: lit.call{{.*}}(%mem)
       var mem = detail_fn()
       # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem
       # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
       mem.noop()
       # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%mem)

# CHECK-LABEL: lit.struct.decl @Parameterized<level: !Int>
struct Parameterized[level: Int]:
    fn __init__(out self): pass

    fn __del__(owned self):
        pass

# CHECK-LABEL: lit.fn @"test_parameterized
fn test_parameterized():
  # CHECK: %x = lit.var.decl "x"
  # expected-warning @+1 {{assignment to 'x' was never used}}
  var x = Parameterized[4]()
  # CHECK: lit.call {{.*}}@"__init__{{.*}}(%x)
  # CHECK: lit.call {{.*}}__del__{{.*}}<:!Int {4}>(%x)

struct Complicated:
  var a: MemExample
  var b: MemExample

# This exercises turning a pop.pointer into an RValue, which produces an 'owned'
# pointer magically from memory.
# CHECK-LABEL: lit.fn @"testTakePointeeAsOwned1
fn testTakePointeeAsOwned1(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`]):
  # This should run the destructor.
  # CHECK-NEXT: [[REF1:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REF1]])
  _ = __get_address_as_owned_value(ptr)

  # This should run the destructor and not get omitted.
  # CHECK-NEXT: [[REF2:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REF2]])
  _ = __get_address_as_owned_value(ptr)

# CHECK-LABEL: lit.fn @"testTakePointeeAsOwned2
fn testTakePointeeAsOwned2(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`],
                          i1ptr: __mlir_type.`!kgen.pointer<i1>`):

  # The RValue can be consumed directly.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[REF]])
  consume(__get_address_as_owned_value(ptr))

  # i1 doesn't have ownership but should still work for generality.
  # CHECK-NEXT: %ownedI1 = lit.var.decl
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %i1ptr end_uninit :
  # CHECK-NEXT: [[I1VAL:%.*]] = lit.load.consume [[REF]]
  # CHECK-NEXT: lifetime.start %ownedI1
  # CHECK-NEXT: lit.ref.store [[I1VAL]], %ownedI1
  # CHECK-NEXT: lifetime.end %ownedI1
  # expected-warning @+1 {{assignment to 'ownedI1' was never used}}
  var ownedI1 = __get_address_as_owned_value(i1ptr)

  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>


# CHECK-LABEL: testGetAsUnitializedObject
fn testGetAsUnitializedObject(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`]):
   # Overwriting the value in a __get_address_as_uninit_lvalue does not destroy
  # the memory, because it is uninit.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr start_uninit :
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])
  __get_address_as_uninit_lvalue(ptr) = MemExample()

  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: testCondGetAsUnitializedObject
# Early exit from fn using __get_address_as_uninit_lvalue should work.
# https://github.com/modularml/modular/issues/27472
fn testCondGetAsUnitializedObject(exit_early: __mlir_type.i1,
                                  ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`]):
  # CHECK: hlcf.elif
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

  # CHECK: lit.fn @"__init__
  fn __init__(out self):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%0)
    self.f1 = MemExample()
    # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f2]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%2)
    self.f2 = MemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return

  # CHECK: lit.fn @"__init__
  fn __init__(out self, a: MemExample, b: MemExample):
    self.f1 = a
    self.f2 = b

  fn __copyinit__(out self, existing: Self):
    self = Self(existing.f1, existing.f2)

  # CHECK-LABEL: lit.fn @"mutate
  fn mutate(mut self):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%0)

    # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%2)
    self.f1 = MemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>

 # CHECK-LABEL: lit.fn @"mutate2
  fn mutate2(mut self):
    # Disable the dtor of 'self' before we overwrite it to show we can do this,
    # both F1 and F2 need to be destroyed before being overwritten
    # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F1R]])
    # CHECK-NEXT: [[F2R:%.*]] = lit.ref.struct.ger %self[f2]
    # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F2R]])

    # CHECK-NEXT: lit.ownership.mark_destroyed %self
    __disable_del self

    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%self)
    self = FieldSensitiveMemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>


  # CHECK-LABEL: lit.fn @"__del__
  # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%0)
  # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f2]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%2)
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: lit.ownership.mark_destroyed %self

# This disables the destructor of 'x' which causes the fields to be destroyed.
# CHECK-LABEL: lit.fn @"disableDtor
fn disableDtor(owned x: FieldSensitiveMemExample):
  # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %x[f1]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F1R]])
  # CHECK-NEXT: [[F2R:%.*]] = lit.ref.struct.ger %x[f2]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F2R]])
  # CHECK-NEXT: lit.ownership.mark_destroyed %x
  # CHECK-NEXT: kgen.param.constant: none
  __disable_del x

# CHECK-LABEL: lit.fn @"regpassable_owned_args_mutable
fn regpassable_owned_args_mutable(owned x: RegExample):
  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x)
  x.mutate()

  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%x)
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}"__init__{{.*}}()
  # CHECK-NEXT: lit.ref.store [[TMP]], %x
  x = RegExample()

  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x)
  x.mutate()
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%x)

# Result optimization cannot emit directly into a value that is passed as an
# argument, because this forms a mutable reference to something immutable
# implicitly.  We must invoke the copy ctor.
# CHECK-LABEL: lit.fn @"use_and_return
fn use_and_return(a: FieldSensitiveMemExample) -> FieldSensitiveMemExample:
  # This will read from 'a' and write into the result slot in an arbitrary
  # order. They cannot alias.
  return FieldSensitiveMemExample(a.f2, a.f1)

fn use_and_return2(a: FieldSensitiveMemExample) -> MemExample:
  return a.f2

fn use_inout_and_return(mut a: FieldSensitiveMemExample) -> FieldSensitiveMemExample:
  return a

fn return_ref(mut a: FieldSensitiveMemExample) -> ref [a] FieldSensitiveMemExample:
  return a

# CHECK-LABEL: lit.fn @"test_result_optimization
fn test_result_optimization():
  # CHECK-NEXT: %example = lit.var.decl "example"
  # CHECK-NEXT: lifetime.start %example
  # CHECK-NEXT: lit.call @{{.*}}"__init__{{.*}}(%example)
  var example = FieldSensitiveMemExample()

  # Direct reuse of the result slot forces a temporary.

  # CHECK: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
  # CHECK-NEXT: lifetime.start %__call_result_tmp__
  # CHECK-NEXT: lit.call {{.*}}use_and_return{{.*}}([[IMMREF]], %__call_result_tmp__)
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%example)
  # CHECK-NEXT: lifetime.end %example
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %__call_result_tmp__
  # CHECK-NEXT: lifetime.start %example
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[IMMREF]], %example)
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%__call_result_tmp__)
  # CHECK-NEXT: lifetime.end %__call_result_tmp__
  example = use_and_return(example)

  # Aliased reuse of part of the result slot forces a temporary.

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: [[F1:%.*]] = lit.ref.struct.ger %example[f1]
  # CHECK-NEXT: %__call_result_tmp___0 = lit.var.decl
  # CHECK-NEXT: lifetime.start %__call_result_tmp___0
  # CHECK-NEXT: lit.call @ownership::@"use_and_return2{{.*}}([[IMMREF]], %__call_result_tmp___0)
  example.f1 = use_and_return2(example)
  # CHECK-NEXT: [[F1_2:%.*]] = lit.ref.struct.ger %example[f1]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F1_2]])
  # CHECK-NEXT: lit.call @{{.*}}@"__moveinit__{{.*}}(%__call_result_tmp___0, [[F1]])
  # CHECK-NEXT: lifetime.end %__call_result_tmp___0

  # Mutating self through a reference forces a temporary.
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: [[RETREF:%.*]] = lit.call {{.*}}return_ref{{.*}}(%example)
  # CHECK-NEXT: [[TMPVAR:%.*]] = lit.var.decl
  # CHECK-NEXT: lifetime.start [[TMPVAR]]
  # CHECK-NEXT: lit.call {{.*}}use_and_return{{.*}}([[IMMREF]], [[TMPVAR]])
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[TMPVAR]]

  # Delete the old thing at the reference pointed-to-by return_ref before we
  # copy into it.
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[RETREF]])

  # FieldSensitiveMem doesn't have a moveinit.
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[IMMREF]], [[RETREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[TMPVAR]])
  # CHECK-NEXT: lifetime.end [[TMPVAR]]
  return_ref(example) = use_and_return(example)

  # CHECK-NEXT: lit.call @{{.*}}@"mutate{{.*}}(%example)
  example.mutate()
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%example)
  # CHECK-NEXT: lifetime.end %example

  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>

# CHECK-LABEL: lit.fn @"impl_mutable_arg
def impl_mutable_arg(a: FieldSensitiveMemExample, mut b: FieldSensitiveMemExample) -> None:
  # CHECK-NEXT: lit.call {{.*}}@"__del__{{.*}}(%b)
  # CHECK-NEXT: %a_0 = lit.var.decl "a" arg(0)
  # CHECK-NEXT: lifetime.start %a_0
  # CHECK-NEXT: lit.call {{.*}}@"__copyinit__{{.*}}(%a, %a_0)
  # CHECK-NEXT: lit.call {{.*}}use_inout_and_return{{.*}}(%a_0, %b)
  # CHECK-NEXT: lit.call {{.*}}@"__del__{{.*}}(%a_0)

  b = use_inout_and_return(a)

##===----------------------------------------------------------------------===##
# Consume Expressions
##===----------------------------------------------------------------------===##

# CHECK: lit.fn @"test_result_consume_reg
fn test_result_consume_reg(cond: __mlir_type.i1) -> RegExample:
  # CHECK-NEXT: %example2 = lit.var.decl
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: lit.var.lifetime.start %example2
  # CHECK-NEXT: lit.ref.store [[TMP]], %example2
  var example2 = RegExample()

  # CHECK-NEXT: hlcf.elif
  # CHECK-NEXT: hlcf.elif.yield
  # CHECK-NEXT: } then {
  if (cond):
    # CHECK-NEXT: lit.ownership.use %example2
    # CHECK-NEXT: [[TMP2:%.*]] = lit.load.consume %example2
    # CHECK-NEXT: lifetime.end %example2
    # CHECK-NEXT: kgen.return [[TMP2]]
    return example2^
  else: # CHECK-NEXT: } else {
    # CHECK-NEXT: [[TMP2:%.*]] = lit.load.consume %example2
    # CHECK-NEXT: lifetime.end %example2
    # CHECK-NEXT: kgen.return [[TMP2]]
    return example2  # copy/del -> move optimization.

# CHECK: lit.fn @"consumeMem
fn consumeMem(owned x: MemExample):
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%x)
  # CHECK-NEXT: kgen.param.constant: none
  pass

# CHECK: lit.fn @"test_result_consume_mem
fn test_result_consume_mem(cond: __mlir_type.i1) -> MemExample:
  # CHECK-NEXT: %example = lit.var.decl
  # CHECK-NEXT: lifetime.start %example
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%example)
  var example = MemExample()

  # This doesn't consume example, so it must copy it. It does consume the copy.
  # CHECK-NEXT: %anonymous2A = lit.var.decl
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: lifetime.start %anonymous2A
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], %anonymous2A)
  # CHECK-NEXT: lit.call {{.*}}consumeMem{{.*}}(%anonymous2A)
  # CHECK-NEXT: lifetime.end %anonymous2A
  consumeMem(example)

  # This does consume example, so no copy needed.
  # CHECK-NEXT: lit.ownership.use %example
  # CHECK-NEXT: lit.call {{.*}}consumeMem{{.*}}(%example)
  # CHECK-NEXT: lifetime.end %example
  consumeMem(example^)

  # CHECK-NEXT: %example2 = lit.var.decl
  # CHECK-NEXT: lifetime.start %example2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%example2)
  var example2 = MemExample()

  # CHECK-NEXT: lit.ownership.use %example2
  # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%example2, %__result__)
  # CHECK-NEXT: lifetime.end %example2
  # CHECK-NEXT: kgen.param.constant: none
  return example2^

# CHECK-LABEL: lit.struct.decl @BigRegExample
@register_passable
struct BigRegExample:
  var a: RegExample
  var b: RegExample

  # CHECK-LABEL: lit.fn @"__init__()"
  fn __init__(out self):
    # CHECK-NEXT: %self = lit.var.decl "self" initoutarg
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: [[A:%.*]] = lit.ref.struct.ger %self[a]
    # CHECK-NEXT: lit.ref.store [[TMP]], [[A]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: [[B:%.*]] = lit.ref.struct.ger %self[b]
    # CHECK-NEXT: lit.ref.store [[TMP]], [[B]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %self
    # CHECK-NEXT: lit.var.lifetime.end %self
    # CHECK-NEXT: kgen.return [[TMP]]
    self.a = RegExample()
    self.b = RegExample()

  # CHECK-LABEL: lit.fn @"__copyinit__
  fn __copyinit__(out self, existing: Self):
    # CHECK-NEXT: %self = lit.var.decl "self" initoutarg
    # CHECK-NEXT: [[EA:%.*]] = lit.ref.struct.ger %existing[a]
    # CHECK-NEXT: [[SA:%.*]] = lit.ref.struct.ger %self[a]
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[EA]])
    # CHECK-NEXT: lit.ref.store [[TMP]], [[SA]]
    # CHECK-NEXT: [[EB:%.*]] = lit.ref.struct.ger %existing[b]
    # CHECK-NEXT: [[SB:%.*]] = lit.ref.struct.ger %self[b]
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[EB]])
    # CHECK-NEXT: lit.ref.store [[TMP]], [[SB]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %self
    # CHECK-NEXT: lit.var.lifetime.end %self
    # CHECK-NEXT: kgen.return [[TMP]]
    self.a = existing.a
    self.b = existing.b

  # CHECK-LABEL: lit.fn @"__del__
  # CHECK-NEXT: [[APTR:%.*]] = lit.ref.struct.ger %self[a]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[APTR]])
  # CHECK-NEXT: [[BPTR:%.*]] = lit.ref.struct.ger %self[b]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BPTR]])
  # CHECK-NEXT:  = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT: lit.ownership.mark_destroyed %self
  # CHECK-NEXT: kgen.return


fn take_regexample_ref(ref r: RegExample): pass
fn ret_big_reg() -> BigRegExample:
  return BigRegExample()

# CHECK-LABEL: lit.fn @"bigreg_test
fn bigreg_test():
  # CHECK-NEXT: %varThing = lit.var.decl "varThing"
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: lifetime.start %varThing
  # CHECK-NEXT: lit.ref.store [[TMP]], %varThing
  var varThing = BigRegExample()

  # CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: lit.ownership.use [[FIELD]]
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[FIELD]])
  consume(varThing.a^)

  # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %varThing[b]
  # CHECK-NEXT: [[BVAL:%.*]] = lit.ref.immut [[BREF]]
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[BVAL]])
  # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: lifetime.start [[ANON]]
  # CHECK-NEXT: lit.ref.store [[TMP]], [[ANON]]
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[ANON]])
  # CHECK-NEXT: lifetime.end [[ANON]]
  consume(varThing.b)

  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: [[AREF:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: lit.ref.store [[TMP]], [[AREF]]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%varThing)
  # CHECK-NEXT: lifetime.end %varThing
  # expected-warning @+1 {{assignment to 'varThing.a' was never used}}
  varThing.a = RegExample()

  # Must drop the value in a register to pass by-ref
  # CHECK-NEXT: [[TMPREG:%.*]] = lit.call {{.*}}ret_big_reg
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lit.var.lifetime.start [[TMP]]
  # CHECK-NEXT: lit.ref.store [[TMPREG]], [[TMP]]
  # CHECK-NEXT: [[ELT:%.*]] = lit.ref.struct.ger [[TMP]][a]
  # CHECK-NEXT: [[ELTIMM:%.*]] = lit.ref.immut [[ELT]]
  # CHECK-NEXT: lit.call {{.*}}take_regexample_ref{{.*}}([[ELTIMM]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[TMP]])
  # CHECK-NEXT: lit.var.lifetime.end [[TMP]]
  take_regexample_ref(ret_big_reg().a)

  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: lit.struct.decl @ExoticDelExample
@register_passable
struct ExoticDelExample:
  var cond: __mlir_type.i1
  var b: BigRegExample
  var c: RegExample

 # CHECK-LABEL: lit.fn @"__del__
  fn __del__(owned self):
    # self.b gets destroyed ASAP since it isn't used.
    # CHECK-NEXT: [[BPTR:%.*]] = lit.ref.struct.ger %self[b]
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BPTR]])

    # Test the condition
    # CHECK-NEXT: hlcf.elif
    # CHECK-NEXT: [[CONDPTR:%.*]] = lit.ref.struct.ger %self[cond]
    # CHECK-NEXT: [[CONDVAL:%.*]] = lit.ref.load [[CONDPTR]]
    # CHECK-NEXT: hlcf.elif.yield [[CONDVAL]]
    # CHECK-NEXT: } then {
    if self.cond:
      # This side we manually consume for c.

      # CHECK-NEXT: [[CREF:%.*]] = lit.ref.struct.ger %self[c]
      # CHECK-NEXT: lit.ownership.use [[CREF]]
      # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[CREF]])
      # CHECK-NEXT: hlcf.yield
      consume(self.c^)

    # CHECK-NEXT: } else {
    # Destroy C automatically on the else side.
    # CHECK-NEXT:  [[CPTR:%.*]] = lit.ref.struct.ger %self[c]
    # CHECK-NEXT:  lit.call {{.*}}__del__{{.*}}([[CPTR]])
    # CHECK-NEXT:  hlcf.yield
    # CHECK-NEXT:}

    # CHECK-NEXT: = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT:lit.ownership.mark_destroyed %self
    # CHECK-NEXT:kgen.return


# CHECK-LABEL: lit.fn @"def_borrowed
# CHECK-SAME: %a: !lit.ref<!MemExample, imm {{.*}}> read_mem
def def_borrowed(a: MemExample) -> None:
  # CHECK: lit.ref.store %none, %__result__
  # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  # CHECK-NEXT: return [[FALSE]]
  pass


# https://github.com/modularml/modular/issues/24161
@register_passable("trivial")
struct AddrSpace:
    var _value: __mlir_type.index
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        self._value = value
    fn value(self) -> __mlir_type.index:
        return self._value
@value
@register_passable("trivial")
struct MemExamplePtr[addrspace: AddrSpace = __mlir_attr.`0:index`]:
    var value: __mlir_type[
        `!kgen.pointer<`, MemExample, `, `, addrspace._value, `>`
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

# CHECK-LABEL: lit.fn @"destruct_generic_return
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
    fn __del__(owned self):
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
  fn __init__(out self):
    self.x = RegExample()

  # Make sure the synthesized dtor is taken register style.
  # CHECK: lit.fn @"__del__{{.*}}(%self: !lit.ref<!RegExampleValue
  # CHECK-NEXT: lit.ref.struct.ger %self[x]
  # CHECK-NEXT: lit.call {{.*}}__del__
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK: lit.ownership.mark_destroyed %self

# [Bug] __result__ is uninitialized
# https://github.com/modularml/modular/issues/27792
# CHECK-LABEL: lit.fn @"test_or
fn test_or(a: MemExample) -> MemExample:
  # CHECK: hlcf.if {{.*}} {
  # CHECK:   lit.call {{.*}}__copyinit__{{.*}}(%a, {{.*}})
  # CHECK: } else {
  # CHECK:   lit.call {{.*}}__copyinit__{{.*}}(%a, {{.*}})
  # CHECK: }
  return a or a


# ===----------------------------------------------------------------------=== #
# Variadics
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"variadic_mems
# CHECK-SAME: [imm *"mems`"](
# CHECK-SAME: %mems: !kgen.variadic<!lit.ref<!MemExample, imm *"mems`">, read_mem> pos_vararg)
fn variadic_mems(*mems: MemExample):
  # CHECK-NEXT: %mems_0 = lit.var.decl
  # CHECK-NEXT: lifetime.start %mems_0
  # CHECK-NEXT: lit.call {{.*}}@VariadicListMem::@"__init__
  # CHECK-SAME: <{{.*}}:!AnyType #MemExample{{.*}}origin<0> = *"mems`"}, :!Bool {:i1 0}>(%mems, %mems_0)
  pass

# CHECK-LABEL: lit.fn @"call_variadic_mems
fn call_variadic_mems(a: MemExample, b: MemExample):
  # CHECK-NEXT: %0 = kgen.rebind %a : !lit.ref<!MemExample, imm *"a`"> to !lit.ref<!MemExample, imm {*"a`", *"b`1"}>
  # CHECK-NEXT: %1 = kgen.rebind %b : !lit.ref<!MemExample, imm *"b`1"> to !lit.ref<!MemExample, imm {*"a`", *"b`1"}>
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.create [%0, %1]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[imm {*"a`", *"b`1"}]([[VAR]])
  variadic_mems(a, b)

  # Variadic use keeps the memory value alive.
  # CHECK-NEXT: %c = lit.var.decl "c"
  # CHECK-NEXT: lifetime.start %c
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%a, %c)
  var c = a
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.splat 1, [[IMMREF]]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[muttoimm *"c`2"]([[VAR]])
  variadic_mems(c)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%c)
  # CHECK-NEXT: lifetime.end %c
  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: lit.fn @"variadic_field_sensitivity
fn variadic_field_sensitivity():
  # Test that we field sensitively track variadics.
  # CHECK:  %memPair = lit.var.decl
  var memPair = MemPair()

  # CHECK: [[AREF:%.*]] = lit.ref.struct.ger %memPair[a]
  # CHECK-NEXT: lit.ownership.use [[AREF]]
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[AREF]])
  _ = memPair.a^  # Destroy a.

  # Can still pass b through varargs.
  # CHECK: [[BREF:%.*]] = lit.ref.struct.ger %memPair[b]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[BREF]]
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.splat 1, [[IMMREF]]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[muttoimm *"memPair`"->b]([[VAR]])
  variadic_mems(memPair.b)

  # Need to restore 'a' so memPair may destruct.
  # CHECK: [[AREF:%.*]] = lit.ref.struct.ger %memPair[a]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[AREF]])
  # expected-warning @+1 {{assignment to 'memPair.a' was never used}}
  memPair.a = MemExample()

  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%memPair)
  # CHECK-NEXT: lifetime.end %memPair
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.fn @"variadic_inout_mems
# CHECK-SAME: [mut *"mems`"](
# CHECK-SAME: %mems: !kgen.variadic<!lit.ref<!MemExample, mut *"mems`">, mut> pos_vararg)
fn variadic_inout_mems(mut *mems: MemExample):
  # CHECK-NEXT: %mems_0 = lit.var.decl
  # CHECK-NEXT: lifetime.start %mems_0
  # CHECK-NEXT: lit.call {{.*}}@VariadicListMem::@"__init__
  # CHECK-SAME: <:!Bool {:i1 1}, :!AnyType #MemExample{{.*}}origin<1> = *"mems`"}, :!Bool {:i1 0}>(%mems, %mems_0)
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mems_0 :
  # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[IMMREF]], [[ZERO]])
  # CHECK-NEXT: [[XREF:%.*]] = lit.ref.struct.ger [[REF]][x]
  # CHECK-NEXT: [[ONE:%.*]] = kgen.param.constant
  # CHECK-NEXT: lit.call {{.*}}__iadd__{{.*}}([[XREF]], [[ONE]])
  # CHECK-NEXT: lifetime.end %mems_0
  mems[0].x += 1

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.fn @"call_variadic_inout_mems
fn call_variadic_inout_mems():
  var a = MemExample()
  var b = MemExample()
  # CHECK: [[AR:%.*]] = kgen.rebind %a : !lit.ref<!MemExample, mut *"a`"> to !lit.ref<!MemExample, mut {*"a`", *"b`1"}>
  # CHECK-NEXT: [[BR:%.*]] = kgen.rebind %b : !lit.ref<!MemExample, mut *"b`1"> to !lit.ref<!MemExample, mut {*"a`", *"b`1"}>
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.create [[[AR]], [[BR]]]
  # CHECK-NEXT: lit.call {{.*}}variadic_inout_mems{{.*}}[mut {*"a`", *"b`1"}]([[VAR]])
  variadic_inout_mems(a, b)

  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[AR]])
  # CHECK-NEXT: lifetime.end %a
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BR]])
  # CHECK-NEXT: lifetime.end %b

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return


# CHECK-LABEL: lit.fn @"test_partial_overwrite
fn test_partial_overwrite(cond: __mlir_type.i1):
  # CHECK-NEXT: %pair = lit.var.decl "pair"
  # CHECK-NEXT: lifetime.start %pair
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%pair)
  var pair = MemPair()

  # CHECK-NEXT: hlcf.elif
  # CHECK-NEXT: hlcf.elif.yield %cond
  # CHECK-NEXT: } then {
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
    # CHECK-NEXT: lifetime.end %pair
    # CHECK-NEXT: hlcf.yield
  else: # CHECK-NEXT: } else {
    # Inserted destruction of whole pair.
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%pair)
    # CHECK-NEXT: lifetime.end %pair

    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return
    return
  # CHECK-NEXT: }

# CHECK-LABEL: lit.struct.decl @UninitField
struct UninitField:
  var field: MemExample

  # CHECK: lit.fn @"__init__()"
  fn __init__(out self):
      # Show that we can mark a field as intentionally uninitialized.
      # Even after checklifetimes, we don't want the thing initialized.
      __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self.field))

      # CHECK-NEXT: %0 = lit.ref.struct.ger %self[field]
      # CHECK-NEXT: lit.ownership.mark_initialized %0
      # CHECK-NEXT: %none = kgen.param.constant
      # CHECK-NEXT: kgen.return %none

fn maybeMemExample() raises -> MemExample:
   return MemExample()

struct HasMemExample:
  var fh: MemExample
  # CHECK-LABEL: lit.fn @"destroyPotentiallyOverwrittenValueRegardlessOfOutcome
  fn destroyPotentiallyOverwrittenValueRegardlessOfOutcome(mut self):
    # CHECK-NEXT: %__try_error__ = lit.var.dec
    # CHECK-NEXT: lit.try {
    try:
      # CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %self[fh]
      # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
      # CHECK-NEXT: lifetime.start %__try_error__
      # CHECK-NEXT: lifetime.start %__call_result_tmp__
      # CHECK-NEXT: lit.call {{.*}}maybeMemExample{{.*}}(%__try_error__, %__call_result_tmp__)
      self.fh = maybeMemExample()

      # Handle the error and other case.  The error isn't used, so delete it
      # here.

      # CHECK-NEXT: if
      # CHECK-NEXT:   lit.call {{.*}}Error::@"__del__{{.*}}(%__try_error__)
      # CHECK-NEXT:   lifetime.end %__try_error__
      # CHECK-NEXT:   mark_consumed %__call_result_tmp__
      # CHECK-NEXT:   lifetime.end %__call_result_tmp__
      # CHECK-NEXT:   lit.try.raise
      # CHECK-NEXT: } else {

      # On success, we overwrite the field.
      # CHECK-NEXT:   [[FIELD2:%.*]] = lit.ref.struct.ger
      # CHECK-NEXT:   lit.call {{.*}}__del__{{.*}}([[FIELD2]])
      # CHECK-NEXT:   mark_consumed %__try_error__
      # CHECK-NEXT:   lifetime.end %__try_error__
      # CHECK-NEXT:   yield
      # CHECK-NEXT: }

      # On success we move the result value into the destination.
      # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%__call_result_tmp__, [[FIELD]])
      # CHECK-NEXT: lifetime.end %__call_result_tmp__
      # CHECK-NEXT: lit.try.yield
    except:
      pass

@value
struct Dim:
  var dim: Int

fn maybeDim() raises -> Dim:
   return Dim(3)

@value
struct List:
  fn append(self, d: MemExample):
     pass

@value
struct DoNotPropagateErrorStateIntoContinueSet:
  var dims: List
  # CHECK-LABEL: lit.fn @"__init__(
  fn __init__(out self, cond: __mlir_type.`i1`, owned list: List) raises:
    # CHECK:     hlcf.loop "_loop_0" {
    # CHECK-NEXT:  hlcf.if %cond {
    # CHECK-NEXT:    hlcf.yield
    # CHECK-NEXT:  } else {
    # CHECK-NEXT:    hlcf.break "_loop_0"
    # CHECK-NEXT:  }
    while cond:
      list.append(maybeMemExample())
    self.dims = list

fn use(x: MemExample): pass

# CHECK-LABEL: lit.fn @"destroyWholeValuesIfLastReferenceWasInLoop
fn destroyWholeValuesIfLastReferenceWasInLoop(cond: __mlir_type.`i1`,
                                              owned memPair: MemPair):
   # Part of mempair is used in the loop, but this keeps the entire thing
   # alive during the loop.  The solution here is to destroy memPair immediately
   # before the implicit break out of the loop
   while cond:
     # CHECK:      hlcf.if %cond {
     # CHECK-NEXT:   hlcf.yield
     # CHECK-NEXT: } else {
     # CHECK-NEXT:   lit.call @{{.*}}::@MemPair::@"__del__({{.*}}(%memPair)
     # CHECK-NEXT:   hlcf.break "_loop_0"
     # CHECK-NEXT: }
     if cond:
        use(memPair.a)

# CHECK-LABEL: lit.fn @"overwrite
# MOCO-700
fn overwrite(y: MemExample, x: Bool) raises:
   var foo = MemPair()
   if x:
   # CHECK: } then {
   # CHECK-NEXT: lit.call @{{.*}}::@MemPair::@"__del__
      raise Error()
   # CHECK: } else {
   # CHECK-NEXT: [[V7:%.*]] = lit.ref.struct.ger %foo[a]
   # CHECK-NEXT: lit.call @{{.*}}@MemExample::@"__del__
   # CHECK-NEXT: hlcf.yield
   # CHECK-NEXT: }
   # CHECK: lit.call @{{.*}}::@MemPair::@"__del__{{.*}}(%foo)
   # expected-warning @+1 {{assignment to 'foo.a' was never used; assign to '_' instead?}}
   foo.a = MemExample()


# CHECK-LABEL: lit.fn @"test_if_ownership
# MOCO-721: Test that ownership is transfered and all the move optimizations are
# done.
fn test_if_ownership(x: Bool, owned a: RegExample, owned b: RegExample) -> RegExample:
    # CHECK-NEXT: lit.call {{.*}}__mlir_i1__
    # CHECK-NEXT: [[RES:%.*]] = hlcf.if
    # CHECK-NEXT:    [[TMP:%.*]] = kgen.rebind %a
    # CHECK-NEXT:    hlcf.yield [[TMP]]
    # CHECK-NEXT:  } else {
    # CHECK-NEXT:    [[TMP:%.*]] = kgen.rebind %b
    # CHECK-NEXT:    hlcf.yield [[TMP]]{{.*}}
    # CHECK-NEXT:  }

    # Copy into a local temporary.
    # CHECK-NEXT:  [[IRES:%.*]] = lit.ref.immut [[RES]]
    # CHECK-NEXT:  [[RESULT:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[IRES]])

    # Last use of both x and b.
    # CHECK-NEXT:    lit.call {{.*}}__del__{{.*}}(%a)
    # CHECK-NEXT:    lit.call {{.*}}__del__{{.*}}(%b)

    # CHECK-NEXT:  kgen.return [[RESULT]]
    return a if x else b


struct MyStructWithMarkDestroyed[T: Copyable & Movable]:
    var a: T
    var b: T

# CHECK-LABEL: lit.fn @{{.*}}reap
    fn reap(owned self, out result: T):
        # "a" field is never used here so it is destroyed early.
        # CHECK-NEXT: [[AREF:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT: lit.call{{.*}}__del__{{.*}}([[AREF]]

        # Transfer operator includes a lit.ownership.use.
        # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %self[b]
        # CHECK-NEXT: lit.ownership.use [[BREF]]

        # Rvalue can be moved into the result slot.
        # CHECK-NEXT: lit.call{{.*}}__moveinit__{{.*}}([[BREF]], %result)
        result = self.b^

        # Full object bit is explicitly destroyed.
        # CHECK-NEXT: lit.ownership.mark_destroyed %self
        __disable_del self

        # CHECK-NEXT: kgen.param.constant: none
        # CHECK-NEXT: kgen.return


# CHECK-LABEL: lit.fn @"field_sensitive_ref_last_use
fn field_sensitive_ref_last_use(owned write_state : IntAndOptional):
    # should destroy ALL OF write_state after the copy into msg.
    var msg = write_state.error.value()

    _ = msg.__len__()

    # CHECK: lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %msg)
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%write_state)

@value
struct IntAndOptional:
    var handle: Int
    var error: Optional[String]


# CHECK: lit.fn @"caught_eh_cleanup
fn caught_eh_cleanup():
    # CHECK-NEXT: %eh1 = lit.var.decl "eh1"
    # CHECK-NEXT: lit.try {
    try:
      # CHECK-NEXT: [[NORMALRESULT:%.*]] = lit.var.decl

      # CHECK: lit.var.lifetime.start %eh1
      # This function raises, potentially defining %eh1.
      _ = maybeDim()
      # CHECK: [[RAISE:%.*]] = lit.call @ownership::@"maybeDim

      # Check for the error and handle it.
      # CHECK-NEXT: hlcf.if [[RAISE]] {
      # EH is never used, so it can be immediately released.
      # CHECK-NEXT:    lit.call {{.*}}__del__{{.*}}(%eh1)
      # CHECK-NEXT:    lit.var.lifetime.end %eh1

      # Normal result is never used
      # CHECK-NEXT: lit.ownership.mark_consumed [[NORMALRESULT]]
      # CHECK-NEXT: lit.var.lifetime.end [[NORMALRESULT]]
      # CHECK-NEXT: lit.try.raise
    except eh1:
      pass

    # CHECK: %eh2 = lit.var.decl "eh2"
    # CHECK-NEXT: lit.try {
    try:
      # CHECK-NEXT: [[NORMALRESULT:%.*]] = lit.var.decl

      # CHECK: lit.var.lifetime.start %eh2
      # This function raises, potentially defining %eh2.
      _ = maybeDim()
      # CHECK: [[RAISE:%.*]] = lit.call @ownership::@"maybeDim

      # Check for the error and handle it.
      # CHECK-NEXT: hlcf.if [[RAISE]] {
      # Normal result is never used
      # CHECK-NEXT: lit.ownership.mark_consumed [[NORMALRESULT]]
      # CHECK-NEXT: lit.var.lifetime.end [[NORMALRESULT]]
      # CHECK-NEXT: lit.try.raise

    # CHECK: } except {
    except eh2:
      # CHECK-NEXT: [[EH2:%.*]] = lit.ref.immut %eh2
      # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[EH2]])
      eh2.use()

    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%eh2)
    # CHECK-NEXT: lit.var.lifetime.end %eh2

# CHECK-LABEL: lit.fn @"test_ref_field
# https://linear.app/modularml/issue/MOCO-1251
fn test_ref_field(owned mem: MemPair):
  # Pointer to subfield.
  r = Pointer(to=mem.a)

  # Subfield reference keeps entire value alive.
  # CHECK: lit.call {{.*}}__eq__
  _ = r == r
  # CHECK-NEXT: lit.call {{.*}}MemPair::@"__del__

fn use_inner_pointer(ptr: UnsafePointer[UInt8]): pass

# CHECK-LABEL: lit.fn @"handleAnyLifetime1
fn handleAnyLifetime1():
  str = String()
  # Make sure this keeps alive str until after the call.
  # CHECK: lit.call {{.*}}use_inner_pointer
  use_inner_pointer(str.unsafe_ptr())
  # CHECK-NEXT: lit.call {{.*}}String::@"__del__{{.*}}(%str)
  # CHECK-NEXT: lit.var.lifetime.end %str

# CHECK-LABEL: lit.fn @"handleAnyLifetime2
fn handleAnyLifetime2():
  ui8 = UInt8()

  # Make sure this keeps 'ui8' alive until after the call even though
  # the element is trivial.
  # CHECK: lit.call {{.*}}use_inner_pointer
  use_inner_pointer(UnsafePointer.address_of(ui8))
  # CHECK-NEXT: lit.var.lifetime.end %ui8

# CHECK-LABEL: lit.fn @"handleAnyLifetime3
fn handleAnyLifetime3():
    # CHECK-NEXT: %a_packed_ptr = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}__init__
    # CHECK-NEXT: lit.var.lifetime.start %a_packed_ptr
    # CHECK-NEXT: lit.ref.store
    # CHECK-NEXT: lit.var.lifetime.end %a_packed_ptr
    # expected-warning @+1 {{assignment to 'a_packed_ptr' was never used}}
    var a_packed_ptr = UnsafePointer[Int]()

    # CHECK-NEXT: lit.call {{.*}}__init__
    # CHECK-NEXT: lit.var.lifetime.start %a_packed_ptr
    # CHECK-NEXT: lit.ref.store
    # CHECK-NEXT: lit.var.lifetime.end %a_packed_ptr

    # This shouldn't be treated as a use of `a_packed_ptr`
    # expected-warning @+1 {{assignment to 'a_packed_ptr' was never used}}
    a_packed_ptr = UnsafePointer[Int]()


fn take_pack[*Ts: AnyType](*values: *Ts): pass

# CHECK-LABEL: lit.fn @"handleAnyLifetime4
# VariadicPack's need to extend the lifetime in the pack
# https://github.com/modular/mojo/issues/3559
fn handleAnyLifetime4():
  str = String()
  ptr = UnsafePointer.address_of(str)

  # Should extend the lifetime of 'str'.
  take_pack(ptr)

  # CHECK: lit.call {{.*}}take_pack
  # CHECK: lit.call {{.*}}__del__{{.*}}(%str)
  # CHECK: lit.var.lifetime.end %str


struct A:
    var data: UnsafePointer[Int]
    fn __init__(out self):
        self.data = UnsafePointer[Int]()
    fn __del__(owned self): pass

fn use_int(a: Int): pass

# CHECK-LABEL: lit.fn @"handleAnyLifetime5
fn handleAnyLifetime5():
    # lit.ref.load needs to extend the lifetime of A.
    a = A()
    # CHECK: [[INT_REF:%.*]] = {{.*}}UnsafePointer::@"__getitem__
    # CHECK-NOT: lit.call {{.*}}__del__
    # CHECK: lit.ref.load [[INT_REF]]
    # CHECK: lit.call {{.*}}__del__{{.*}}(%a)
    use_int(a.data[0])


# This checks that the Mojo parser successfully folds the initializer call for
# origin into a struct attr, which is important for lifetime analysis to be able
# to reason about these.

# CHECK-LABEL: lit.fn @"test_origin_ctor_folding
fn test_origin_ctor_folding[orig1: Origin[_]](abcdef: A):
    # CHECK-NEXT: lit.alias.decl {{.*}} = <{_mlir_origin: origin<0> = *"abcdef`1"}>
    alias x = Origin(__origin_of(abcdef))

    # MOCO-1467: Origin type equality problem.
    # CHECK-NEXT: lit.alias.decl {{.*}} = <orig1>
    alias y = Origin(orig1._mlir_origin)

    # Check that __origin_of works on origins as well as MValues.
    # CHECK-NEXT: lit.alias.decl *"o2{{.*}} = <{{{.*}}abcdef{{.*}}orig1
    alias o2 = __origin_of(orig1, abcdef)

fn useMemory(a: MemExample): pass

# CHECK-LABEL: lit.fn @"testConds1
fn testConds1(cond: __mlir_type.i1, reg: RegExample, i: Int):
  # Implicit conversions.
  # Mojo Issue #49: https://github.com/modular/mojo/issues/49

  # CHECK-NEXT: hlcf.if %cond -> !RegExample {
  # CHECK:        [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%reg)
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = reg if cond else i

  # CHECK: hlcf.if %cond -> !RegExample {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK:        [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%reg)
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = i if cond else reg

  _ = reg
  _ = i

# Memory only conds. Issue (#13379)
# CHECK-LABEL: lit.fn @"testConds2
fn testConds2(cond: __mlir_type.i1, a: MemExample, b: MemExample) -> MemExample:
  # CHECK:      [[IF:%.*]] = hlcf.if %cond
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind %a
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind %b
  # CHECK-NEXT:   hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }
  # CHECK-NEXT: lit.call {{.*}}useMemory{{.*}}([[IF]])
  useMemory(a if cond else b)

  # Handle a local temp correctly.
  # TODO(ternary memory optimization): The moveinit doesn't seem necessary,
  # could direct construct into the dest and elide the temp.

  # CHECK-NEXT: [[IF:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: hlcf.if %cond
  # CHECK-NEXT:   [[TMP:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT:   lit.var.lifetime.start [[TMP]]
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}([[TMP]])
  # CHECK-NEXT:   lit.var.lifetime.start [[IF]]
  # CHECK-NEXT:   lit.call {{.*}}__moveinit__{{.*}}([[TMP]], [[IF]])
  # CHECK-NEXT:   lit.var.lifetime.end [[TMP]]
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.var.lifetime.start [[IF]]
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%b, [[IF]])
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  # CHECK-NEXT: [[IFI:%.*]] = lit.ref.immut [[IF]]
  # CHECK-NEXT: lit.call {{.*}}useMemory{{.*}}([[IFI]])
  useMemory(MemExample() if cond else b)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[IF]])
  # CHECK-NEXT: lit.var.lifetime.end [[IF]]


  # CHECK-NEXT: [[IF:%.*]] = hlcf.if %cond
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind %a
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind %b
  # CHECK-NEXT:   hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}([[IF]], %__result__)
  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
  return a if cond else b

# CHECK-LABEL: lit.fn @"testConds3
fn testConds3(cond: __mlir_type.i1, owned a: MemExample, owned b: MemExample,
              owned m: RegExample, owned n: RegExample):
  # CHECK-NEXT: %t1 = lit.var.decl
  # CHECK-NEXT: [[IF:%.*]] = hlcf.if %cond
  # CHECK-NEXT:    lit.ownership.use %a
  # CHECK-NEXT:    [[TMP:%.*]] = kgen.rebind %a
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:    lit.ownership.use %b
  # CHECK-NEXT:    [[TMP:%.*]]  = kgen.rebind %b
  # CHECK-NEXT:    hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }
  # CHECK-NEXT: [[IFI:%.*]] = lit.ref.immut [[IF]]
  # CHECK-NEXT: lit.var.lifetime.start %t1
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IFI]], %t1)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%b)
  var t1 = a^ if cond else b^

  # CHECK-NEXT: %t2 = lit.var.decl
  # CHECK-NEXT: [[IF:%.*]] = hlcf.if %cond
  # CHECK-NEXT:    lit.ownership.use %m
  # CHECK-NEXT:    [[TMP:%.*]] = kgen.rebind %m
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:    lit.ownership.use %n
  # CHECK-NEXT:    [[TMP:%.*]]  = kgen.rebind %n
  # CHECK-NEXT:    hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }
  # CHECK-NEXT: [[IFI:%.*]] = lit.ref.immut [[IF]]
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[IFI]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%m)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%n)
  # CHECK-NEXT: lit.var.lifetime.start %t2
  # CHECK-NEXT: lit.ref.store [[TMP]], %t2
  var t2 = m^ if cond else n^

  consume(t1^)
  consume(t2^)

# CHECK-LABEL: lit.fn @"my_min1
# CHECK-SAME: !lit.ref<!Int, mut=and(*"x_is_mut`", *"y_is_mut`2"), {(mutcast mut=*"x_is_mut`", *"x_is_origin`1"), (mutcast mut=*"y_is_mut`2", *"y_is_origin`3")}>
fn my_min1(cond: __mlir_type.i1, ref x: Int, ref y: Int) -> ref [x, y] Int:
  # CHECK-NEXT: [[IF:%.*]] = hlcf.if %cond
  # CHECK-NEXT:    [[TMP:%.*]] = kgen.rebind %x
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:    [[TMP:%.*]]  = kgen.rebind %y
  # CHECK-NEXT:    hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }

  # CHECK-NEXT: kgen.return [[IF]]
  return x if cond else y

# CHECK-LABEL: lit.fn @"my_min2
fn my_min2[T: AnyType](ref a: T, ref b: T) -> ref [a, b] T:
    return a

# CHECK-LABEL: lit.fn @"test_min2
# https://github.com/modular/mojo/issues/3815
fn test_min2(a: String):
    # CHECK: lit.call {{.*}}String::@"__init__
    var x = String()
    # CHECK: lit.call {{.*}}String::@"__init__
    var y = String()
    # CHECK: [[REF:%.*]] = lit.call {{.*}}my_min2
    # CHECK-NEXT: [[SLICE:%.*]] = lit.call {{.*}}StringSlice::@"__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call {{.*}}String::@"__iadd__{{.*}}([[REF]], [[SLICE]])
    my_min2(x, y) += a
    # CHECK-NEXT: lit.call {{.*}}String::@"__del__{{.*}}(%x)
    # CHECK-NEXT: lit.var.lifetime.end %x
    # CHECK-NEXT: lit.call {{.*}}String::@"__del__{{.*}}(%y)
    # CHECK-NEXT: lit.var.lifetime.end %y

# MOCO-1500: Can't take origin of read-only String arg
def origin_of_def_arg(a: String):
    _ = __origin_of(a)

# MOCO-1542: Need to rebind field type when checking size.
@value
struct MyParameterizedField[T: Copyable & Movable]:
  var a: T
  var b: T

fn use_parameterized_field():
  var s = MyParameterizedField[Dim](Dim(8), Dim(3))
  var litref = __get_mvalue_as_litref(s.b)
  var rebind = __mlir_op.`kgen.rebind`
    [_type=Pointer[Int, __origin_of(s.b)]._mlir_type](litref)
  # expected-warning @+1 {{assignment to 'mvalue' was never used}}
  var mvalue = __get_litref_as_mvalue(rebind)

# MOCO-1558: Failure handling sub-type elements
# CHECK-LABEL: OuterStruct
struct OuterStruct:
    var outers_field: RefResultStruct
    # CHECK: lit.fn @"__del__
    fn __del__(owned self):
        # CHECK: lit.call {{.*}}use(::String)
        use(self.outers_field.x)
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[outers_field]
        # CHECK-NEXT: lit.call {{.*}}RefResultStruct::@"__del__{{.*}}([[TMP]])
        # CHECK: lit.ownership.mark_destroyed %self

struct RefResultStruct:
  var x: String
  fn __init__(out self):
    self.x = String()

  fn method(self) -> ref [self.x] String:
      return self.x

fn use(a: String): pass

# https://github.com/modular/modular/issues/4163
#  BUG] Mojo compiler error when two instance variables of type PythonObject are initialized by Python.import_module in a struct's __init__()
struct SomeStruct:
    var test_agent: SomeValue[Int]
    fn __init__(out self) raises:
        self.test_agent = SomeValue(123)

struct SomeValue[T: Copyable & Movable]:
    var value: T
    var name: String
    var tmp: Int

    fn __init__(out self, value: T) raises:
        self.value = value
        self.name = "example"
        self.tmp = 1 #<- remove this field and it works
