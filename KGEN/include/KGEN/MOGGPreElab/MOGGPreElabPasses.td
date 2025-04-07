//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SliceMOGGFuncs
//===----------------------------------------------------------------------===//

def OutlineKernels : Pass<"mogg-outline-kernels", "ModuleOp"> {
  let summary = "Outline MOGG kernel functions";
  let description = [{
    Separate the kernel entry point from the implementation details of the
    kernel by outlining them into a function invoked by the entry point.

    This additionally will mark the kernel as elementwise if it is elementwise

    The outlining will produce one of two forms. The first form is for the case
    of outlining generic kernels. The other form is for the special case of
    elementwise.
    ```
    kgen.generator MY_KERNEL(...) {
      // Any fusion lambdas
      kgen.param.region.declare InLambda(...) {

      }

      kgen.call "MY_KERNEL_OUTLINED"<...InLambda...>(...)
    }
    ```

    For elementwise we retain the elementwise loop but outline the body.
    ```
    kgen.generator MY_KERNEL(...) {
      // Any fusion lambdas
      kgen.param.region.declare InLambda(...) {

      }

      kgen.param.region.declare BODY (...) {
        kgen.call "MY_KERNEL_OUTLINED"<...InLambda...>(...)
      }
      kgen.call "for_each"<body: BODY>(...)
    }
    ```
  }];
}

def AnnotateKernels : Pass<"mogg-annotate-kernels", "ModuleOp"> {
  let summary = "Strips MOGG decorators and replaces them with attributes";
  let description = [{
    A MOGG domain specific pass which will annotate each kernel at the lit
    level allowing subsequent passes to use the information parsed in a
    sanitized way.
  }];
}

def VerifyKernels : Pass<"mogg-verify-kernels", "ModuleOp"> {
  let summary = "Verifies that kernels are well constructed";
  let description = [{
    This pass ensures that the kernels present in the module are well
    constructed. Among other things, it checks that shape functions match
    execute functions and that inputs appear after outputs in the function's
    signature.
  }];
}

def MOGGPreElabPipeline : Pass<"mogg-pre-elab-pipeline", "ModuleOp"> {
  let summary = "Run all the mogg pre elab passes";
  let description = [{
    Run the pipeline of MOGG PreElaborated functions.
  }];
  let options = [
    Option<"debugBuild", "debugBuild", "bool", "false",
           "True when the build is a debug build. Will disable some passes">
  ];
}
