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

def MOGGAutospecialize : Pass<"mogg-autospecialize", "ModuleOp"> {
  let summary = "Add a spec parameter to each tensor on kernels and propagate them";
  let description = [{
    Tensors are often times associated with static values like shape which the
    code can then specialize on and generate the most efficient kernels.
    However, in the context of graph compilation, it only makes sense to
    specialize on shape if you actually have a model with associated activation
    shape information. Finally, including static values as parameters in the
    kernel itself can become cumbersome. If a kernel takes in 5 tensors and each
    of has 5 static fields, then we have 25 parameters to potentially include!

    Instead the user writes code using an unspecialized tensor, and this code
    is then replaced with a specialized version. The call graph is then fixed
    up to call the specialized function.

    E.g.

    fn foo[type: DType, rank: Int](input: ManagedTensorSlice[type, rank]):
      alias static_shape = specsof[type, rank]("input").shape
      ...

    Will become something analogous to:

    # Note the parameterization on `static_spec_input` which is associated with
    # `input`.
    fn foo_1[static_spec_input: StaticTensorSpec, type: DType, rank: Int](
      input: ManagedTensorSlice[type, rank]
    ):
      alias static_spec_input = static_spec.shape
      ...

    Old calls to `foo` will be properly replaced to calls to `foo_1` *if* it is
    on a call graph from rooted from a registered Extensibility API kernel.
  }];
}
