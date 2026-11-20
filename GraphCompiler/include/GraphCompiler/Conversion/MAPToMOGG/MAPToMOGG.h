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

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_H

#include "mlir/Pass/Pass.h"

namespace M::MAP {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h.inc"

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_H
