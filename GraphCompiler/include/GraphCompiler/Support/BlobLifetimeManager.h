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

#ifndef GENERICML_SUPPORT_BLOBLIFETIMEMANAGER_H
#define GENERICML_SUPPORT_BLOBLIFETIMEMANAGER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"

namespace M::GML {

/// Lets buffers take ownership of a resource blob and free it early.
struct BlobLifetimeManager {
  BlobLifetimeManager(
      llvm::unique_function<void(void *data, size_t size, size_t align)>
          deleter,
      llvm::ArrayRef<char> data, size_t alignment)
      : deleter(std::move(deleter)), data(data), alignment(alignment) {}
  ~BlobLifetimeManager() {
    deleter(const_cast<char *>(data.data()), data.size(), alignment);
  }
  llvm::unique_function<void(void *data, size_t size, size_t align)> deleter;
  llvm::ArrayRef<char> data;
  size_t alignment;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_BLOBLIFETIMEMANAGER_H
