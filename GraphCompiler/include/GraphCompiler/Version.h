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

#ifndef GENERICML_GRAPHCOMPILER_VERSION_H
#define GENERICML_GRAPHCOMPILER_VERSION_H

#include <string>

namespace M::MO {

/// Get a unique string that embeds the binary version of the graph compiler
/// implementation.
std::string getGraphCompilerVersionString();

/// Get a hex hash that embeds the binary version of the MOGG compiler
/// implementation.
std::string getMOGGCompilerVersionString();

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_VERSION_H
