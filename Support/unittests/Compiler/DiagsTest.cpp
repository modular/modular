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

#include "Support/Compiler/Diags.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace M;
using namespace mlir;

namespace {

/// Collects the text of each message that reaches the MLIR diagnostic engine.
class DiagsTest : public ::testing::Test {
protected:
  DiagsTest()
      : diags(sourceMgr, &context, /*useMLIRDiagnostics=*/true,
              /*maxNotesPerDiagnostic=*/10, /*stripFilenamePrefix=*/"",
              /*disableWarnings=*/false, /*warningsAsErrors=*/false),
        handler(&context, [this](Diagnostic &diag) {
          messages.push_back(diag.str());
          for (Diagnostic &note : diag.getNotes())
            messages.push_back(note.str());
          return mlir::success();
        }) {}

  MLIRContext context;
  llvm::SourceMgr sourceMgr;
  Diags diags;
  ScopedDiagnosticHandler handler;
  std::vector<std::string> messages;
  Location loc = UnknownLoc::get(&context);
};

TEST_F(DiagsTest, LazyTextIsFormattedOnEmission) {
  int numFormatted = 0;
  {
    InflightDiag diag = diags.emitError(loc, "expected ");
    diag.addLazyText([&] {
      ++numFormatted;
      return std::string("'A'");
    });
    diag << ", got ";

    // Concatenating a diagnostic keeps its lazy text in place.
    InflightDiag reason = diags.emitError(loc, "type ");
    reason.addLazyText([&] {
      ++numFormatted;
      return std::string("'B'");
    });
    reason << " here";
    diag << std::move(reason);

    diag.attachNote(loc) << "see ";
    diag.addLazyText([&] {
      ++numFormatted;
      return std::string("'C'");
    });
    EXPECT_EQ(numFormatted, 0);
  }
  EXPECT_EQ(numFormatted, 3);
  ASSERT_EQ(messages.size(), 2u);
  EXPECT_EQ(messages[0], "expected 'A', got type 'B' here");
  EXPECT_EQ(messages[1], "see 'C'");
}

TEST_F(DiagsTest, AbandonedDiagnosticDoesNotFormatLazyText) {
  int numFormatted = 0;
  {
    InflightDiag diag = diags.emitError(loc, "expected ");
    diag.addLazyText([&] {
      ++numFormatted;
      return std::string("'A'");
    });
    diag.abandon();
  }
  EXPECT_EQ(numFormatted, 0);
  EXPECT_TRUE(messages.empty());
}

} // namespace
