// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file -verify-diagnostics %s

lit.fn @shared_var() {
  // expected-error @+1 {{variables outside the default address space are not yet supported}}
  %a = lit.var.decl "a" var : !lit.ref<index, mut *"a", 3>
  hlcf.return
}
