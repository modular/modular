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

#ifndef MLRT_CORE_IO_CUFILE_H
#define MLRT_CORE_IO_CUFILE_H

#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include <fcntl.h>
#include <filesystem>
#include <unistd.h>

namespace M::IO {
/// A wrapper around the NVIDIA GPUDirect Storage (GDS) cuFile API
/// for performing direct I/O between storage and GPU memory.
struct CUFile {
  /// Initializes the cuFile library and driver resources. Must be called before
  /// any other cuFile API function. This would also register the uninitializer
  /// for you.
  static ErrorOrSuccess init();

  /// Registers a device memory buffer with cuFile for direct DMA transfers.
  ErrorOrSuccess registerMemory(void *devPtr, size_t size);

  /// Deregisters a previously registered device memory buffer.
  ErrorOrSuccess deregisterMemory(void *devPtr);

  /// Opens a file and registers it with cuFile.
  ErrorOrSuccess open(const std::filesystem::path &path, int flags = O_RDONLY);

  /// Deregisters the file handle from cuFile and closes the file descriptor.
  ErrorOrSuccess close();

  /// Reads data directly from the file into a registered device buffer.
  ErrorOr<ssize_t> read(void *devPtr, size_t size, off_t fileOffset,
                        off_t devOffset);

  /// Writes data directly from a registered device buffer to the file.
  ErrorOr<ssize_t> write(void *devPtr, size_t size, off_t fileOffset,
                         off_t devOffset);

private:
#if MODULAR_LINUX
  /// The underlying file descriptor.
  int fd = -1;
  /// The cuFile handle for the registered file.
  void *handle = nullptr;
#endif // MODULAR_LINUX
};
}; // namespace M::IO

#endif // MLRT_CORE_IO_CUFILE_H
