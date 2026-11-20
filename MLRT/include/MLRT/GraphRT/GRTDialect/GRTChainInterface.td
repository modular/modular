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
// An interface for chains.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTCHAININTERFACE_TD
#define MLRT_GRAPHRT_GRTDIALECT_GRTCHAININTERFACE_TD

include "mlir/IR/OpBase.td"

def GRT_ChainCreateOpInterface : OpInterface<"GRTChainCreateOpInterface"> {
  let description = [{
    Operation that creates a new chain.
  }];
  let cppNamespace = "::M::GRT";
  let methods = [
    InterfaceMethod<
      "Get the resulting chain",
      "::mlir::Value", "getChainResult", (ins), "return $_op.getResult();"
    >,
    InterfaceMethod<
      "Get the input chains that this chain depends on",
      "::mlir::ValueRange", "getInputChains", (ins)
    >
  ];
}

def GRT_GatedOnChainOpInterface : OpInterface<"GRTGatedOnChainOpInterface"> {
  let description = [{
    Chain operations that depend on other chains/inputs. Allows changing what
    is being depended on. Ops implementing this interface must have the same
    type input chains as output chains.
  }];
  let cppNamespace = "::M::GRT";
  let methods = [
    InterfaceMethod<
      "Set the inputs that this chain depends on",
      "void", "setInputChains", (ins "::mlir::ValueRange":$inChains)
    >
  ];
}

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTCHAININTERFACE_TD
