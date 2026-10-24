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

#include "GraphCompiler/Support/Telemetry.h"
#include "Support/Preprocessor.h"

using namespace M;
using namespace GML;

std::string M::GML::stringifyErrorCode(MaxEngineErrorCode code) {

  auto prependStr = [](StringRef string) {
    return "max.engine.failure." + string.str();
  };
  switch (code) {
  case M::GML::MaxEngineErrorCode::UserInputError:
    return prependStr("user_input_error");
  case M::GML::MaxEngineErrorCode::FileNotFoundError:
    return prependStr("file_not_found_error");
  case M::GML::MaxEngineErrorCode::UnknownFormatError:
    return prependStr("unknown_format_error");
  case M::GML::MaxEngineErrorCode::TFImportError:
    return prependStr("tf_import_error");
  case M::GML::MaxEngineErrorCode::TorchImportError:
    return prependStr("torch_import_error");
  case M::GML::MaxEngineErrorCode::GraphLoweringError:
    return prependStr("graph_lowering_error");
  case M::GML::MaxEngineErrorCode::MissingOpError:
    return prependStr("missing_op_error");
  case M::GML::MaxEngineErrorCode::CompilationError:
    return prependStr("compilation_error");
  }
}
