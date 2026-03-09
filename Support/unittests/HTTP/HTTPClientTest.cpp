//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HTTP/HTTPClient.h"
#include "Support/FileSystemExtras.h"
#include "llvm/Support/Threading.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>

using namespace M;

namespace {

constexpr StringLiteral testData =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit";

} // namespace

TempFile createTempFile() {
  std::error_code ec;
  std::filesystem::path tempDir = std::filesystem::temp_directory_path(ec);
  EXPECT_FALSE(ec);
  auto temp =
      TempFile::create((tempDir / "modular_payload_%%%%%%.txt").string());

  EXPECT_FALSE(temp.isError());
  // Open the file in output mode. If the file doesn't exist, it will be created
  std::ofstream outfile(temp->getPath());

  EXPECT_EQ(outfile.is_open(), true);
  outfile << testData.data();
  outfile.close(); // Remember to close the file when done
  return temp.takeValue();
}
