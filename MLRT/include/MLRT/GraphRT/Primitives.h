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
// This file declares the interfaces to register the primitives in the GraphRT
// library.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_PRIMITIVES_H
#define MLRT_GRAPHRT_PRIMITIVES_H

namespace M {
class PrimitiveRegistry;

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void registerGraphRTTestPrimitives(PrimitiveRegistry &registry);

/// This registers the 'graphrt' dialect primitives for working with integers.
void registerGraphRTIntegerPrimitives(PrimitiveRegistry &registry);

/// This registers the 'graphrt' dialect primitives for control flow.
void registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry);

/// This registers all the primitive libraries in the GraphRT subsystem.
inline void registerAllGraphRTPrimitives(PrimitiveRegistry &registry) {
  registerGraphRTTestPrimitives(registry);
  registerGraphRTIntegerPrimitives(registry);
  registerGraphRTControlFlowPrimitives(registry);
}

} // namespace M

#endif // MLRT_GRAPHRT_PRIMITIVES_H
