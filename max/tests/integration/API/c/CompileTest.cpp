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
//
// Compile tests
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace M;

TEST_F(APITest, CompileErrorAsync) {
  M_CompileConfig *compileConfig = M_newCompileConfig();

  M_AsyncCompiledModel *compiledModel =
      M_compileModel(context, &compileConfig, status);
  EXPECT_THAT(compiledModel, ::testing::NotNull());
  EXPECT_SUCCESS(status);
  EXPECT_THAT(compileConfig, ::testing::IsNull())
      << "compilation takes ownership of compile config";

  // Error happens on wait, not on start
  M_waitForCompilation(compiledModel, status);
  EXPECT_FAILURE(status,
                 "Model path or source is not set in compilation config");

  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, CompileErrorNoPathSet) {
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_THAT(compiledModel, ::testing::NotNull());
  EXPECT_THAT(compileConfig, ::testing::IsNull())
      << "compilation takes ownership of compile config";
  EXPECT_FAILURE(status,
                 "Model path or source is not set in compilation config");
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, CompileErrorForInvalidPath) {
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, "/nonexistent/path/to/model.mef");
  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_THAT(compiledModel, ::testing::NotNull());
  EXPECT_THAT(compileConfig, ::testing::IsNull())
      << "compilation takes ownership of compile config";
  EXPECT_FAILURE(status, "invalid path");
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, CompileErrorForInvalidMEFFile) {
  // Create a temporary file with invalid content
  std::string invalidMefPath = testing::TempDir() + "/invalid_test.mef";
  std::ofstream f(invalidMefPath);
  ASSERT_TRUE(f.is_open()) << "Failed to create temp file";
  f << "This is not a valid MEF file";
  f.close();

  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, invalidMefPath.c_str());
  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_THAT(compiledModel, ::testing::NotNull());
  EXPECT_THAT(compileConfig, ::testing::IsNull())
      << "compilation takes ownership of compile config";
  EXPECT_FAILURE(status, "");
  M_freeCompiledModel(compiledModel);
}
