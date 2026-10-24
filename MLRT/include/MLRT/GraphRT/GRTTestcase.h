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

#ifndef MLRT_GRAPHRT_GRTTESTCASE_H
#define MLRT_GRAPHRT_GRTTESTCASE_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

namespace M {
class MEFInvokable;
class PrimitiveDescriptor;

class GRTTestcase {
public:
  /// Provide a constructor from PrimitiveDescriptor.
  static ErrorOr<GRTTestcase>
  getFromDescriptor(const PrimitiveDescriptor &desc) {
    return GRTTestcase(desc);
  }

  /// Enables dyn_cast/cast/isa on this TopLevelOp by providing the name of this
  /// primitive.
  static StringLiteral getEntityName();

  StringRef getName() const { return name; }

  /// Get the body for this op - a single Invokable.
  MEFInvokable *getBody() const { return body; }

private:
  GRTTestcase(const PrimitiveDescriptor &desc);
  GRTTestcase(StringRef name, MEFInvokable *body) : name(name), body(body) {}
  StringRef name;
  MEFInvokable *body;
};
} // namespace M

#endif // MLRT_GRAPHRT_GRTTESTCASE_H
