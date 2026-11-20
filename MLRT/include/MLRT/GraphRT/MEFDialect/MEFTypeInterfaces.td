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
// Interfaces for MEF types which may also be used by other dialects which
// have similar chain types.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFDIALECT_MEFTYPEINTERFACES_TD
#define MLRT_GRAPHRT_MEFDIALECT_MEFTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def MEF_ChainTypeInterface : TypeInterface<"MEFChainTypeInterface"> {
  let description = [{
    Interface for types which support chain-like operations.
  }];
  let cppNamespace = "::M::MEF";
}

def MEF_AnyTypeInterface : TypeInterface<"MEFAnyTypeInterface"> {
  let description = [{
    This interface allows foreign types to be used as model input and output
    types. These types are represented in the runtime by AnyAsyncValueRefs of
    arbitrary type.
  }];
  let cppNamespace = "::M::MEF";
}

#endif  // MLRT_GRAPHRT_MEFDIALECT_MEFTYPEINTERFACES_TD
