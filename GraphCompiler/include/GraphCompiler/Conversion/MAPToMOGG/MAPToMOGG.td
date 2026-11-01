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

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_TD

include "mlir/Pass/PassBase.td"

def MAPToMOGG : Pass<"map-to-mogg", "mlir::ModuleOp"> {
  let summary = "Lower `map` dialect kernels into `mogg` codegen ops.";
  let description = [{
    Lowers `map` dialect kernels into `mogg` codegen ops: dynamic dims become
    SSA values, contexts are added, and tensors are built from pointers. This is
    where the fusion-focused `map` IR crosses into the code-generation-focused
    `mogg` IR.
  }];

  let options = [
    Option<"enableComputeLambdas", "enable-compute-lambdas", "bool",
      /*default=*/"true",
      "Build compute lambdas for epilogues feeding a kernel argument that "
      "declares `_FusedComputeOutputTensor`. When false, such an epilogue gets "
      "a classic store lambda instead.">,
  ];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
    "M::MO::MODialect",
    "M::MOSH::MOSHDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_TD
