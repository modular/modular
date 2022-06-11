// RUN: kgen-opt -allow-unregistered-dialect %s -verify-diagnostics

kgen.generator @test<p1, p2: si64>() {
  
  "someop" () {
    attr1 = #kgen.param.decl.ref<"p" : i4>,
    // TODO: xpected-error @+1 {{bad thing}}
    attr2 = #kgen.param.decl.ref<"p" : i1>
  } : () -> ()

  kgen.return
}

