//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Disabled: MAXPLAT-223
#if 0

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(MOGGAnnotateTests, MissingReadWriteParams) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

@compiler.register("Missing")
struct Missing:
    @staticmethod
    fn execute(a: ManagedTensorSlice):
        ...
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 2);
            EXPECT_EQ(diags[0].message, "Error for argument 'a': 'mut' "
                                        "inferred parameter must be set");
            EXPECT_EQ(diags[1].message, "Error for argument 'a': 'input' "
                                        "inferred parameter must be set");
          })
      .execute();
}

TEST(MOGGAnnotateTests, OutputAfterInput) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

@compiler.register("OutputAfterInput")
struct OutputAfterInput:
    @staticmethod
    fn execute(a: InputTensor, b: OutputTensor):
        ...
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "Output tensor argument 'b' must come before "
                                 "other non-output tensor arguments");
                     })
      .execute();
}

TEST(MOGGAnnotateTests, InputTensorsForShape) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

@compiler.register("InputTensorsForShape")
struct InputTensorsForShape:
    @staticmethod
    fn execute(a: InputTensor):
      pass

    @staticmethod
    fn shape(a: ManagedTensorSlice):
      pass
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "Error for argument 'a': Tensor arguments to "
                                 "shape functions must be 'InputTensor'");
                     })
      .execute();
}

TEST(MOGGAnnotateTests, NonInputTensorList) {
  Document doc("test:///foo.mojo", R"(
from compiler_internal import StaticTensorSpec
import compiler_internal as compiler
from tensor import OutputTensor
from tensor_internal.managed_tensor_slice import _MutableInputTensor as MutableInputTensor

@compiler.register("non_input_tensor_list")
struct NonInputTensorList:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
        _synchronous: Bool,
    ](
        output: List[
            OutputTensor[
                static_spec = StaticTensorSpec[type, rank].create_unknown()
            ]
        ],
    ):
      pass
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(
                           diags[0].message,
                           "Only input tensors are allowed as the element type "
                           "for list arguments at the moment.");
                     })
      .execute();
}

TEST(MOGGAnnotateTests, PytorchFallbackInvalidArgument) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from python import Python, PythonObject

@compiler.register("pytorch_fallback")
struct InvalidPytorchFallBackArgument:
    @staticmethod
    fn execute():
      ...

    @staticmethod
    fn pytorch_fallback(a: PythonObject, b: Int):
      return
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 2);
            EXPECT_EQ(
                diags[0].message,
                "Error for argument 'b' all arguments to 'pytorch_fallback' "
                "functions must have type 'PythonObject'");
            EXPECT_EQ(diags[1].message,
                      "Error for result type: the only permitted return type "
                      "for 'pytorch_fallback' functions is 'PythonObject'");
          })
      .execute();
}

TEST(MOGGAnnotateTests, PytorchFallbackInvalidResult) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from python import Python, PythonObject

@compiler.register("pytorch_fallback")
struct InvalidPytorchFallBackResult:
    @staticmethod
    fn execute():
      ...

    @staticmethod
    fn pytorch_fallback(a: PythonObject, b: PythonObject) -> Int:
      return Int(0)
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 1);
            EXPECT_EQ(diags[0].message,
                      "Error for result type: the only permitted return type "
                      "for 'pytorch_fallback' functions is 'PythonObject'");
          })
      .execute();
}

TEST(MOGGAnnotateTests, PytorchFallbackInvalidOutResult) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from python import Python, PythonObject

@compiler.register("pytorch_fallback")
struct InvalidPytorchFallBackResult:
    @staticmethod
    fn execute():
      ...

    @staticmethod
    fn pytorch_fallback(out output: Int, a: PythonObject, b: PythonObject):
      return Int(0)
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 1);
            EXPECT_EQ(diags[0].message,
                      "Error for result type: the only permitted return type "
                      "for 'pytorch_fallback' functions is 'PythonObject'");
          })
      .execute();
}

#endif // #if 0
