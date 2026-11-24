//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MOJO_DEPGRAPHDIALECT_DEPGRAPHPASSES_H
#define MOJO_DEPGRAPHDIALECT_DEPGRAPHPASSES_H

namespace M::KGEN::DepGraph {

/// Registers the `depgraph` dialect passes with the global pass registry.
void registerDepGraphPasses();

} // namespace M::KGEN::DepGraph

#endif // MOJO_DEPGRAPHDIALECT_DEPGRAPHPASSES_H
