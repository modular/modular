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

#ifndef MLRT_GRAPHRT_MEFEXECUTOR_MEFFILE_H
#define MLRT_GRAPHRT_MEFEXECUTOR_MEFFILE_H

#include "AsyncRT/ForwardDecls.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace M {
class MEFInvokable;
class PrimitiveRegistry;
class PrimitiveDescriptor;

/// Instances of this class represent opened and decoded MEF files, allowing
/// clients to interact with the invokable list, etc.
class MEFFile : public ReferenceCounted<MEFFile> {
public:
  virtual ~MEFFile();

  /// Open and read a MEF file, setting up our internal state and returning a
  /// pointer to our initialized object on success.  On an initial parse
  /// failure, an error message is returned in the `ErrorOr`.  MEF file format
  /// errors discovered lazily when running invokables, so they are returned
  /// through the invokable's result AsyncValue's.
  ///
  /// Note that this doesn't set up the primitive table, it requires an explicit
  /// call to `installPrimitives` to do so.
  ///
  /// TODO: This should (optionally) manage ownership of the underlying data
  /// passed in.  For example, we could take a closure to run when the lifetime
  /// of the MEFFile is done, or have a reference counted abstraction for
  /// holding memory buffers.
  static ErrorOr<RCRef<MEFFile>> open(StringRef file);

  /// Check that the primitives name table is valid and can be resolved to
  /// entries in the specified registry.
  ErrorOrSuccess installPrimitives(const PrimitiveRegistry &registry);

  /// This method inspects the contents of the MEF file to make an educated
  /// guess of what it contains, allowing for better QoI and usability of tools.
  CompiledFrameworkLabel sniffFileContents() const;

  /// Open and read a MEF file, setting up our internal state and returning a
  /// pointer to our initialized object on success.  On an initial parse
  /// failure, an error message is returned in the `ErrorOr`.  MEF file format
  /// errors discovered lazily when running invokables, so they are returned
  /// through the invokable's result AsyncValue's.
  ///
  /// Note that this doesn't set up the primitive table, it requires an explicit
  /// call to `installPrimitives` to do so.
  ///
  /// TODO: This should (optionally) manage ownership of the underlying data
  /// passed in.  For example, we could take a closure to run when the lifetime
  /// of the MEFFile is done, or have a reference counted abstraction for
  /// holding memory buffers.
  static ErrorOr<RCRef<MEFFile>> open(StringRef file,
                                      const PrimitiveRegistry &registry);

  /// Return the top level invokable in the file. This object corresponds to the
  /// MLIR module in the file used to create the MEF.
  const MEFInvokable *getTopLevelInvokable() const;

protected:
  MEFFile() = default;
};

/// Return true if the specified file contents look like it is a MEF file, by
/// checking the magic number.
bool isMEFFile(llvm::StringRef contents);

} // namespace M

#endif // MLRT_GRAPHRT_MEFEXECUTOR_MEFFILE_H
