//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SliceMOGGFuncs
//===----------------------------------------------------------------------===//

def SliceMOGGFuncs : Pass<"slice-mogg-funcs", "mlir::ModuleOp"> {
  let summary = "Slices shape and compute from mogg kernels";
  let description = [{
    A mogg domain specific pass which performs KGEN to KGEN transforms that
    the graph compiler can understand. The KGEN will be legal regardless.

    This description to expand as the functionality is added to the pass.
  }];
}

def OutlineMOGGFuncs : Pass<"mogg-outline", "mlir::ModuleOp"> {
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

def MOGGAnnotate : Pass<"mogg-annotate", "mlir::ModuleOp"> {
  let summary = "Strips MOGG decorators and replaces them with attributes";
  let description = [{
    A mogg domain specific pass which will annotate each kernel at the lit
    level allowing subsequent passes to use the information parsed in a
    sanitized way.
  }];
}

def MOGGPreElabPipeline : Pass<"mogg-pre-elab-pipeline", "mlir::ModuleOp"> {
  let summary = "Run all the mogg pre elab passes";
  let description = [{
    Run the pipeline of MOGG PreElaborated functions.
  }];
  let options = [
    Option<"debugBuild", "debugBuild", "bool", "false",
           "True when the build is a debug build. Will disable some passes">
  ];
}

def MOGGAutoparameterize : Pass<"mogg-autoparameterize", "mlir::ModuleOp"> {
  let summary = "Add a spec parameter to each tensor on kernels and propagate them";
  let description = [{
    This pass clones kernel functions and makes them take a spec parameter for each
    input and rewrites the call graph so that every function called by those tensors
    also takes a spec and gets the right spec.
  }];
}

def MOGGIdentifyKernels : Pass<"mogg-identify-kernels", "mlir::ModuleOp"> {
  let summary = "Identifies kernels and labels them";
  let description = [{
    A mogg domain specific pass which will identify kernels and label execute
    and shape functions (if present).
  }];
}

def MOGGLabelTensorParams : Pass<"mogg-label-tensor-params", "mlir::ModuleOp"> {
  let summary = "Attaches a label to kernel tensors for subsequent use";
  let description = [{
    This will label the parameters of kernel tensors such that subsequent passes
    can find them more easily.
  }];
}
