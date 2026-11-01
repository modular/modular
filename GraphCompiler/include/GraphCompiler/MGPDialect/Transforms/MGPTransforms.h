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

#ifndef GENERICML_MGPDIALECT_MGPTRANSFORMS_H
#define GENERICML_MGPDIALECT_MGPTRANSFORMS_H

#include "Support/AlignedAlloc.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace M::MGP {

#define GEN_PASS_DECL
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

#define GEN_PASS_REGISTRATION
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_H
