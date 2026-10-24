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

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_TD

include "mlir/Pass/PassBase.td"

def MOToMAP : Pass<"mo-to-map", "mlir::ModuleOp"> {
  let summary = "Lower `mo` dialect ops into `map` kernels.";
  let description = [{
    This pass lowers known `mo` dialect ops into one `map` kernel each,
    as the entry point of the advanced fusion pipeline.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::MAP::MAPDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_TD
