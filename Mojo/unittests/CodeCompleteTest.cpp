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

#include "Mojo/MojoTooling/CodeComplete.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/FrontendOptions.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::KGEN;

TEST(CodeCompleteTest, CachedContextOwnsRequestSource) {
  MLIRContext mlirContext{MLIRContext::Threading::DISABLED};
  llvm::SourceMgr sourceMgr;
  CompilationOptions options;
  FrontendOptions frontendOptions;
  frontendOptions.useBuiltinModule = false;
  MojoParserContext parser(sourceMgr, &mlirContext, options, frontendOptions);
  llvm::StringRef expected = R"(struct Target:
    pass

def test():
    Tar)";
  std::string source = expected.str();
  unsigned fileId = 0;
  auto results = Mojo::codeCompleteInContext(
      parser, llvm::MemoryBufferRef(source, "completion.mojo"), source.size(),
      [&](int id) {
        fileId = id;
        parser.parseFileForLSP(id);
      });
  ASSERT_TRUE(llvm::any_of(results, [](const auto &result) {
    return result.label == "Target";
  }));

  source.front() = '!';
  EXPECT_EQ(sourceMgr.getMemoryBuffer(fileId)->getBuffer(), expected);
}
