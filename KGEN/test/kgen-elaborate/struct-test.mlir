// RUN: kgen-opt %s | kgen-opt -o /dev/null

kgen.struct.decl @FooStruct<T:type> {
  // expected-note @below {{previously declared as "x" here}}
  x : !pop.pointer<T>
}

kgen.generator.interface @wrapInFoo<T:type>(!pop.pointer<T>)
    -> !kgen.ref<@FooStruct<T:type = T>>

kgen.generator @wrapInFooImpl<T:type>(%a: !pop.pointer<T>)
    -> !kgen.ref<@FooStruct<T:type = T>>
    implements @wrapInFoo {
  %0 = kgen.struct.create(%a) : (!pop.pointer<T>) -> !kgen.ref<@FooStruct<T:type = T>>
  kgen.return %0 : !kgen.ref<@FooStruct<T:type = T>>
}
