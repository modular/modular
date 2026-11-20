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

#ifndef GENERICML_COMPILEDRIVER_COMPILEDRIVER_H
#define GENERICML_COMPILEDRIVER_COMPILEDRIVER_H

#include "AsyncRT/ForwardDecls.h"
#include "Cache/CachedTransform.h"
#include "Cache/Support/Keys.h"
#include "GraphCompiler/Passes.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/SymbolExport.h"
#include "llvm/Support/Process.h"

namespace M {

MODULAR_CXX_EXPORT std::optional<std::filesystem::path> getMaxCacheDir();

/// Gets the location for storing externalized weights.
std::optional<std::filesystem::path> getWeightsCacheDir();

/// Checks if IR caching is enabled.
bool isModelIRCacheEnabled();

/// Returns the MEF from `loweringFunction`, served from and stored to the cache
/// when caching is enabled.
ErrorOr<BufferRef>
cacheMEFTransform(StringRef frameworkName,
                  llvm::function_ref<ErrorOr<BufferRef>()> loweringFunction,
                  WriteableBufferRef key);

/// Builds a cache key from a file buffer and options, each of which must have
/// a `print(raw_ostream &)` method.
template <typename... Options>
WriteableBufferRef makeCacheKey(BufferRef buffer, Options &...options) {
  WriteableBufferRef keyBuffer = WriteableBuffer::get();

  auto insertOptionToKey = [&](auto &option) { option.print(*keyBuffer); };
  (insertOptionToKey(options), ...);

  std::string bufferKey = Cache::Keys::BufferKey::hashKey(std::move(buffer));
  keyBuffer->write(bufferKey.data(), bufferKey.size());
  return keyBuffer;
}
} // namespace M

#endif // GENERICML_COMPILEDRIVER_COMPILEDRIVER_H
