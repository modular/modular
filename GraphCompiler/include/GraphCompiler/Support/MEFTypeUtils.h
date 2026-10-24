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
// This file declares utilities to work with MEFType and its subclasses in the
// context of GraphCompiler.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_MEFTYPEUTILS_H
#define GENERICML_SUPPORT_MEFTYPEUTILS_H

#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/LLVMForwardDecls.h"

namespace M {

class MEFFunctionType;
class ErrorOrSuccess;
class QuantTensorSpec;

/// Decodes a MEFFunctionType signature into input and output QuantTensorSpecs.
ErrorOrSuccess
decodeQuantizedSignature(MEFFunctionType signature,
                         SmallVectorImpl<QuantTensorSpec> &inputSpecs,
                         SmallVectorImpl<QuantTensorSpec> &outputSpecs);

/// Decodes e.g. {argument_names = ["a", "b"], result_names = ["c"]} into
/// `inputNames` and `outputNames`.
ErrorOrSuccess decodeTensorNames(MEFDictionaryAttr attrDictionary,
                                 SmallVectorImpl<StringRef> &inputNames,
                                 SmallVectorImpl<StringRef> &outputNames);

} // namespace M

#endif // GENERICML_SUPPORT_MEFTYPEUTILS_H
