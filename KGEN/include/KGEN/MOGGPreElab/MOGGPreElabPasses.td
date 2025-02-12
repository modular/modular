//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SliceMOGGFuncs
//===----------------------------------------------------------------------===//

def SliceMOGGFuncs : Pass<"slice-mogg-funcs", "ModuleOp"> {
  let summary = "Slices compute from DPS kernels";
  let description = [{
    A mogg domain specific pass which performs KGEN to KGEN transforms that
    the graph compiler can understand. The KGEN will be legal regardless.
    We use this pass to enable kernel fusion in the Graph Compiler.
    We add input / output / elementwise lambdas that can be manipulated
    easily by the Graph Compiler.

    For every compute kernel, it creates a new cloned kernel
    where the in/out lambda parameters are replaced
    by new lambdas that directly read / write from the kernel I/Os.
    Eg:
    ```
    fn foo[spec: StaticTensorSpec](x: ManagedTensorSlice):
      spec.input_lamda[width](idx)
      bar(spec.shape)
    ```

    becomes:
    ```
    fn foo_COMPUTE[spec: StaticTensorSpec](x: ManagedTensorSlice):
      @parameter
      fn input_lambda[width](idx): return x.load[width](idx)
      alias new_spec = StaticTensorSpec(spec.shape, spec.strides,
                                        input_lambda, output_lambda=None)

      new_spec.input_lamda[width](idx)
      bar(new_spec.shape)
    ```

    The graph compiler will use `foo_COMPUTE` and can fuse kernels by directly
    manipulating these lambda regions.
  }];
}

def OutlineMOGGFuncs : Pass<"mogg-outline", "ModuleOp"> {
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

def MOGGAnnotate : Pass<"mogg-annotate", "ModuleOp"> {
  let summary = "Strips MOGG decorators and replaces them with attributes";
  let description = [{
    A mogg domain specific pass which will annotate each kernel at the lit
    level allowing subsequent passes to use the information parsed in a
    sanitized way.
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

