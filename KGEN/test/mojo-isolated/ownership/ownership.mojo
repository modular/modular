# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

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
# CHECK: destructor {{.*}}@RegExample::@"__del__
@register_passable
struct RegExample:
  fn __init__(inout self):
    return 
  fn __copyinit__(inout self, existing: Self): # CHECK: lit.func @"__copyinit__
    return

  fn noop(self): pass
  # CHECK-LABEL: lit.func @"__del__
  # CHECK-NEXT: %self_0 = lit.var.decl "self" arg
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
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%reg)
  # CHECK-NEXT: [[REG:%.*]] = lit.ref.load %reg
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REG]])

  mem2.noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem2
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)

  # CHECK-NEXT:  %mem3 = lit.var.decl "mem3"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem3)
  var mem3 = MemExample()

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
  someInt = 5  # silence var warning.
  # CHECK-NEXT: = lit.ref.load %someInt
  _ = someInt^

  # CHECK-NEXT: [[REG:%.*]] = kgen.param.materialize: !RegExample
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[REG]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[REG]])
  RegExample{}.noop()

  # CHECK-NEXT: %localReg = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}@RegExample::@"__init__{{.*}}(%localReg)
  # CHECK-NEXT: %anonymous2A_0 = lit.var.decl "anonymous
  # CHECK-NEXT: [[REG:%.*]] = lit.ref.load %localReg
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A_0, [[REG]])
  # CHECK-NEXT: [[REG2C:%.*]] = lit.load.consume %anonymous2A_0
  # CHECK-NEXT: [[REG:%.*]] = lit.ref.load %localReg
  # CHECK-NEXT: [[BIGREG:%.*]] = lit.struct.create(a=[[REG2C]], b=[[REG]])
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BIGREG]])
  var localReg = RegExample()
  _ = BigRegExample{a: localReg, b: localReg }

# CHECK-LABEL: lit.func @"indirect_call
fn indirect_call[detail_fn: fn() -> MemExample]():
       # CHECK: %mem = lit.var.decl
       # CHECK-NEXT: lit.call{{.*}}(%mem)
       var mem = detail_fn()
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
  var x = Parameterized[4]()
  # CHECK: lit.call {{.*}}@"__init__{{.*}}(%x)
  # CHECK: lit.call {{.*}}__del__{{.*}}<:!Int {4}>(%x)

struct Complicated:
  var a: MemExample
  var b: MemExample

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
    # Disable the dtor of 'self' before we overwrite it to show we can do this,
    # both F1 and F2 need to be destroyed before being overwritten
    # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F1R]])
    # CHECK-NEXT: [[F2R:%.*]] = lit.ref.struct.ger %self[f2]
    # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F2R]])

    # CHECK-NEXT: lit.ownership.mark_destroyed %self
    __mlir_op.`lit.ownership.mark_destroyed`[_type=None](
       __get_mvalue_as_litref(self))

    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%self)
    self = FieldSensitiveMemExample()
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
  # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %x[f1]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F1R]])
  # CHECK-NEXT: [[F2R:%.*]] = lit.ref.struct.ger %x[f2]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[F2R]])
  # CHECK-NEXT: lit.ownership.mark_destroyed %x
  # CHECK-NEXT: kgen.param.constant: none
  __mlir_op.`lit.ownership.mark_destroyed`[_type=None](
       __get_mvalue_as_litref(x))

# CHECK-LABEL: lit.func @"regpassable_owned_args_mutable
fn regpassable_owned_args_mutable(owned x: RegExample):
  # CHECK-NEXT: %x_0 = lit.var.decl "x" arg
  # CHECK-NEXT: lit.ref.store %x, %x_0
  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x_0)
  x.mutate()

  # CHECK-NEXT: [[X:%.*]] = lit.ref.load %x_0
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[X]])
  # CHECK-NEXT: lit.call {{.*}}"__init__{{.*}}(%x_0)
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

fn use_inout_and_return(inout a: FieldSensitiveMemExample) -> FieldSensitiveMemExample: 
  return a

fn return_ref(inout a: FieldSensitiveMemExample) -> ref [__lifetime_of(a)] FieldSensitiveMemExample:
  return a

# CHECK-LABEL: lit.func @"test_result_optimization
fn test_result_optimization():
  # CHECK-NEXT: %example = lit.var.decl "example"
  # CHECK-NEXT: lit.call @{{.*}}"__init__{{.*}}(%example)
  var example = FieldSensitiveMemExample()

  # Direct reuse of the result slot forces a temporary.

  # CHECK: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}use_and_return{{.*}}([[IMMREF]], %__call_result_tmp__)
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%example)
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %__call_result_tmp__
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}(%example, [[IMMREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%__call_result_tmp__)
  example = use_and_return(example)

  # Aliased reuse of part of the result slot forces a temporary.

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: [[F1:%.*]] = lit.ref.struct.ger %example[f1]
  # CHECK-NEXT: %__call_result_tmp___0 = lit.var.decl
  # CHECK-NEXT: lit.call @ownership::@"use_and_return2{{.*}}([[IMMREF]], %__call_result_tmp___0)
  example.f1 = use_and_return2(example)
  # CHECK-NEXT: [[F1_2:%.*]] = lit.ref.struct.ger [[IMMREF]][f1]
  # CHECK-NEXT: [[MUTREF:%.*]] = kgen.rebind [[F1_2]]
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[MUTREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"__moveinit__{{.*}}([[F1]], %__call_result_tmp___0)

  # Mutating self through a reference forces a temporary.
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: [[RETREF:%.*]] = lit.call {{.*}}return_ref{{.*}}(%example)
  # CHECK-NEXT: [[TMPVAR:%.*]] = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}use_and_return{{.*}}([[IMMREF]], [[TMPVAR]])
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[TMPVAR]]

  # Delete the old thing at the reference pointed-to-by return_ref before we
  # copy into it.
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[RETREF]])

  # FieldSensitiveMem doesn't have a moveinit.
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[RETREF]], [[IMMREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}([[TMPVAR]])
  return_ref(example) = use_and_return(example)

  # CHECK-NEXT: lit.call @{{.*}}@"mutate{{.*}}(%example)
  example.mutate()
  # CHECK-NEXT: lit.call @{{.*}}@"__del__{{.*}}(%example)

  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>

# CHECK-LABEL: lit.func @"impl_mutable_arg
def impl_mutable_arg(a: FieldSensitiveMemExample, inout b: FieldSensitiveMemExample) -> None:
  # CHECK-NEXT: lit.call {{.*}}@"__del__{{.*}}(%b)
  # CHECK-NEXT: %a_0 = lit.var.decl "a" arg(0)
  # CHECK-NEXT: lit.call {{.*}}@"__copyinit__{{.*}}(%a_0, %a)
  # CHECK-NEXT: lit.call {{.*}}use_inout_and_return{{.*}}(%a_0, %b)
  # CHECK-NEXT: lit.call {{.*}}@"__del__{{.*}}(%a_0)

  b = use_inout_and_return(a)

##===----------------------------------------------------------------------===##
# Consume Expressions
##===----------------------------------------------------------------------===##

# CHECK: lit.func @"test_result_consume_reg
fn test_result_consume_reg(cond: __mlir_type.i1) -> RegExample:
  # CHECK-NEXT: %example2 = lit.var.decl
  # CHECK: lit.call {{.*}}__init__{{.*}}(%example2)
  var example2 = RegExample()

  # CHECK-NEXT: hlcf.elif
  # CHECK-NEXT: hlcf.elif.yield
  # CHECK-NEXT: } then {
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
  var example = MemExample()

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
  var example2 = MemExample()

  # CHECK-NEXT: [[CONSUME:%.*]] = lit.transfer_mem_ownership %example2
  # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%__result__, [[CONSUME]])
  # CHECK-NEXT: kgen.param.constant: none
  return example2^

# CHECK-LABEL: lit.struct.decl @BigRegExample
@register_passable
struct BigRegExample:
  var a: RegExample
  var b: RegExample

  # CHECK-LABEL: lit.func @"__init__(ownership::BigRegExample=&)"
  fn __init__(inout self):
    # CHECK-NEXT: %0 = kgen.rebind %self
    # CHECK-NEXT: %1 = lit.ref.struct.ger %0[a]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%1)
    # CHECK-NEXT: %3 = lit.ref.struct.ger %0[b]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%3)
    self.a = RegExample()
    self.b = RegExample()

  # CHECK-LABEL: lit.func @"__copyinit__
  fn __copyinit__(inout self, existing: Self):
    # CHECK-NEXT: %0 = kgen.rebind %self
    # CHECK-NEXT: %1 = lit.struct.extract %existing[a]
    # CHECK-NEXT: %2 = lit.ref.struct.ger %0[a]
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%2, %1)
    # CHECK-NEXT: %4 = lit.struct.extract %existing[b]
    # CHECK-NEXT: %5 = lit.ref.struct.ger %0[b]
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%5, %4)
    self.a = existing.a
    self.b = existing.b

  # CHECK-LABEL: lit.func @"__del__
  # CHECK-NEXT: %self_0 = lit.var.decl "self" arg
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
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%varThing)
  var varThing = BigRegExample()

  # CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: [[LIFEEND:%.*]] = lit.transfer_mem_ownership [[FIELD]]
  # CHECK-NEXT: [[AVAL:%.*]] = lit.load.consume [[LIFEEND]]
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[AVAL]])
  consume(varThing.a^)

  # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %varThing[b]
  # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: [[BVAL:%.*]] = lit.ref.load [[BREF]]
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[ANON]], [[BVAL]])
  # CHECK-NEXT: [[BCOPY:%.*]] = lit.load.consume [[ANON]]
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[BCOPY]])
  consume(varThing.b)

  # CHECK-NEXT: [[AREF:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[AREF]])
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
    # CHECK-NEXT: %self_0 = lit.var.decl "self" arg
    # CHECK-NEXT: lit.ref.store %self, %self_0

    # self.b gets destroyed ASAP since it isn't used.
    # CHECK-NEXT: [[BPTR:%.*]] = lit.ref.struct.ger %self_0[b]
    # CHECK-NEXT: [[BVAL:%.*]] = lit.ref.load [[BPTR]]
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[BVAL]])

    # Test the condition
    # CHECK-NEXT: hlcf.elif
    # CHECK-NEXT: [[CONDPTR:%.*]] = lit.ref.struct.ger %self_0[cond]
    # CHECK-NEXT: [[CONDVAL:%.*]] = lit.ref.load [[CONDPTR]]
    # CHECK-NEXT: hlcf.elif.yield [[CONDVAL]]
    # CHECK-NEXT: } then {
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
def def_borrowed(a: MemExample) -> None:
  # CHECK: lit.ref.store %none, %__result__
  # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  # CHECK-NEXT: return [[FALSE]]
  pass


# https://github.com/modularml/modular/issues/24161
@register_passable("trivial")
struct AddrSpace:
    var _value: __mlir_type.index
    @always_inline("nodebug")
    fn __init__(inout self, value: __mlir_type.index):
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
  fn __init__(inout self):
    self.x = RegExample()

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
# CHECK-SAME: %mems: !kgen.variadic<!lit.ref<!MemExample, imm *"mems`">, borrow_in_mem> borrow|var)
fn variadic_mems(*mems: MemExample):
  # CHECK-NEXT: %mems_0 = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}@VariadicListMem::@"__init__
  # CHECK-SAME: <:!AnyType #MemExample{{.*}}:lifetime<0> *"mems`">(%mems_0, %mems)
  pass

# CHECK-LABEL: lit.func @"call_variadic_mems
fn call_variadic_mems(a: MemExample, b: MemExample):
  # CHECK-NEXT: %0 = kgen.rebind %a : !lit.ref<!MemExample, imm *"a`"> to !lit.ref<!MemExample, imm {*"a`", *"b`1"}>
  # CHECK-NEXT: %1 = kgen.rebind %b : !lit.ref<!MemExample, imm *"b`1"> to !lit.ref<!MemExample, imm {*"a`", *"b`1"}>
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.create [%0, %1]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[imm {*"a`", *"b`1"}]([[VAR]])
  variadic_mems(a, b)

  # Variadic use keeps the memory value alive.
  # CHECK-NEXT: %c = lit.var.decl "c"
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%c, %a)
  var c = a
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.splat 1, [[IMMREF]]
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[muttoimm *"c`2"]([[VAR]])
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
  # CHECK-NEXT: lit.call {{.*}}variadic_mems{{.*}}[muttoimm *"memPair`"]([[VAR]])
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
# CHECK-SAME: %mems: !kgen.variadic<!lit.ref<!MemExample, mut *"mems`">, inout> borrow|var)
fn variadic_inout_mems(inout *mems: MemExample):
  # CHECK-NEXT: %mems_0 = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}@VariadicListMem::@"__init__
  # CHECK-SAME: <:!AnyType #MemExample{{.*}} :lifetime<1> *"mems`">(%mems_0, %mems)
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mems_0 :
  # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[IMMREF]], [[ZERO]])
  # CHECK-NEXT: [[XREF:%.*]] = lit.ref.struct.ger [[REF]][x]
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
  # CHECK: [[AR:%.*]] = kgen.rebind %a : !lit.ref<!MemExample, mut *"a`"> to !lit.ref<!MemExample, mut {*"a`", *"b`1"}>
  # CHECK-NEXT: [[BR:%.*]] = kgen.rebind %b : !lit.ref<!MemExample, mut *"b`1"> to !lit.ref<!MemExample, mut {*"a`", *"b`1"}>
  # CHECK-NEXT: [[VAR:%.*]] = pop.variadic.create [[[AR]], [[BR]]]
  # CHECK-NEXT: lit.call {{.*}}variadic_inout_mems{{.*}}[mut {*"a`", *"b`1"}]([[VAR]])
  variadic_inout_mems(a, b)

  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%b)
  # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return


# CHECK-LABEL: lit.func @"test_partial_overwrite
fn test_partial_overwrite(cond: __mlir_type.i1):
  # CHECK-NEXT: %pair = lit.var.decl "pair"
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
    # CHECK-NEXT: hlcf.yield
  else: # CHECK-NEXT: } else {
    # Inserted destruction of whole pair.
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%pair)

    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return
    return
  # CHECK-NEXT: }

# CHECK-LABEL: UninitField
struct UninitField:
  var field: MemExample

  # CHECK: lit.func @"__init__
  fn __init__(inout self):
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
  # CHECK: lit.func @"destroyPotentiallyOverwrittenValueRegardlessOfOutcome
  fn destroyPotentiallyOverwrittenValueRegardlessOfOutcome(inout self):
    # CHECK-NEXT: %__try_error__ = lit.var.dec
    # CHECK-NEXT: lit.try {
    try:
      # CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %self[fh]
      # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
      # CHECK-NEXT: lit.call {{.*}}maybeMemExample{{.*}}(%__try_error__, %__call_result_tmp__)
      self.fh = maybeMemExample()

      # Handle the error and other case.  The error isn't used, so delete it
      # here.

      # CHECK-NEXT: hlcf.if
      #   CHECK-NEXT: lit.ownership.mark_initialized %__try_error__
      #   CHECK-NEXT: lit.ref.load %__try_error__
      #   CHECK-NEXT: lit.call {{.*}}Error::@"__del__
      #   CHECK-NEXT: lit.try.raise
      # CHECK-NEXT: } else {

      # On success, we overwrite the field.
      # CHECK-NEXT: [[FIELD2:%.*]] = lit.ref.struct.ger %self[fh]
      # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[FIELD2]])
      # CHECK-NEXT: lit.ownership.mark_initialized %__call_result_tmp__
      # CHECK-NEXT: hlcf.yield
      # CHECK-NEXT: }

      # On success we move the result value into the destination.
      # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}([[FIELD]], %__call_result_tmp__)
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
  # CHECK-LABEL: @"__init__({{.*}}::DoNotPropagateErrorStateIntoContinueSet
  fn __init__(inout self, cond: __mlir_type.`i1`, owned list: List) raises:
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

# CHECK-LABEL: lit.func @"destroyWholeValuesIfLastReferenceWasInLoop
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

# CHECK-LABEL: lit.func @"overwrite
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
   foo.a = MemExample()


# CHECK-LABEL: lit.func @"test_if_ownership
# MOCO-721: Test that ownership is transfered and all the move optimizations are
# done.
fn test_if_ownership(x: Bool, owned a: RegExample, owned b: RegExample) -> RegExample:
    # CHECK-NEXT: %a_0 = lit.var.decl
    # CHECK-NEXT: lit.ref.store %a, %a_0
    # CHECK-NEXT: %b_1 = lit.var.decl
    # CHECK-NEXT: lit.ref.store %b, %b_1


    # CHECK-NEXT: lit.call {{.*}}__mlir_i1__
    # CHECK-NEXT: [[RES:%.*]] = hlcf.if
    # CHECK-NEXT:    [[B:%.*]] = lit.ref.load %b_1
    # CHECK-NEXT:    lit.call {{.*}}__del__{{.*}}([[B]])
    # CHECK-NEXT:    [[A:%.*]] = lit.ref.load %a_0
    # CHECK-NEXT:    hlcf.yield [[A]]
    # CHECK-NEXT:  } else {
    # CHECK-NEXT:    [[A:%.*]] = lit.ref.load %a_0
    # CHECK-NEXT:    lit.call {{.*}}__del__{{.*}}([[A]])
    # CHECK-NEXT:    [[B:%.*]] = lit.ref.load %b_1
    # CHECK-NEXT:    hlcf.yield [[B]]
    # CHECK-NEXT:  }
    # CHECK-NEXT:  kgen.return [[RES]]
    return a if x else b

