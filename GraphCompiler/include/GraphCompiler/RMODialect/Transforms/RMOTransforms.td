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
//
// This file declares the transforms for the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def LegalizeRMOOps : Pass<"mo-legalize-rmo-operators", "GraphOp"> {
  let summary = "Legalize RMO operators to mo operators.";
  let description = [{
    `MOAnalogue` ops lower automatically to their MO op by rebinding each
    operand and result to the types from `inferMOTypes`. Other RMO ops lower to
    compositions of RMO ops.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::RMO::RMODialect"
  ];

  let options = [
    Option<"skipRMORebind", "skip-rmo-rebinds", "bool", "false",
           "Whether to skip legalize rmo rebind operations. Useful for debugging.">,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_TD
