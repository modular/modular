//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOGGTOMGP_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOGGTOMGP_TD

include "mlir/Pass/PassBase.td"

def MOGGToMGP : Pass<"mogg-to-mgp", "ModuleOp"> {
  let summary = "Convert `mo` dialect to primitives.";
  let description = [{
    This pass performs a conversion from known `mo` dialect ops to `mgp` and
    `mef` primitives.
  }];
  let options = [
    Option<"compiledFrameworkLabel", "compiled-framework-label", "std::string",
      /*default=*/"\"mof\"",
      "The label for resulting MGP graph. Default is mof.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">
  ];
  let dependentDialects = ["GRT::GRTDialect",
                           "KGEN::POP::POPDialect",
                           "MEF::MEFDialect",
                           "MGP::MGPDialect",
                           "MOGG::MOGGDialect",
                           "mlir::index::IndexDialect"];
}


def LowerPrimitives: Pass<"mgp-lower-primitives", "MGP::ModelOp"> {
  let summary = "Lower MGP operations to primitives";
  let description = [{
    Using the loaded kernel library this looks for implementations for all
    primitives and if one exists will lower the primitive to that mojo
    implementation.

    At this stage these implementations will use "unpacked" synchronous values,
    not the runtime async values.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">
  ];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect",
                          "M::KGEN::KGENDialect",
                          "M::KGEN::LIT::LITDialect",
                          "M::KGEN::POP::POPDialect",
                          "M::HLCF::HLCFDialect",
                          "M::DebugInfo::DebugInfoDialect",
                          "mlir::NVVM::NVVMDialect",
                          "mlir::ROCDL::ROCDLDialect",
                          "mlir::index::IndexDialect",
  ];
}


def RaiseBackIndexConstant: Pass<"mgp-raise-back-index-constant", "ModuleOp"> {
  let summary = "Recover index constants if not part of kernel deduplication";
  let description = [{
    If the lowering of index.constant (mgp.buffer.constant + mgp.buffer._to_index)
    did not help deduplication of kernel stubs, raise those operations back to
    an index.constant if there is only one call to the kernel stub.
  }];

  let dependentDialects = ["MGP::MGPDialect"];
}


#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOGGTOMGP_TD
