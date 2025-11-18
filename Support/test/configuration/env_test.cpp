//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// This tool exposes the getter API's of `M::Config` class from
// `Support/Configuration.h` to enable testing the behavior of this
// foundational component.

#include "Support/CommandLine.h"
#include "Support/Configuration.h"

#include <cstdlib>
#include <iostream>

using namespace M;

namespace {

struct ConfigurationEnvTestCLIOptions {
  cl::opt<bool> ModularConfigFolderPath{
      "ModularConfigFolderPath",
      cl::desc("M::Config::getModularConfigFolderPath()"),
  };

  cl::opt<bool> ModularDataFolderPath{
      "ModularDataFolderPath",
      cl::desc("M::Config::getModularConfigFolderPath()"),
  };

  cl::opt<bool> ConfigFilePath{
      "ConfigFilePath",
      cl::desc("M::Conifg::getConfigFilePath()"),
  };
};

} // namespace

int main(int argc, char **argv) {
  ConfigurationEnvTestCLIOptions cli;
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    R"(Configuration Environmnet Test Tool

Multiple options may be specified at once.

Output is in JSON format and consists of a dictionary where the keys are the
name of the command-line option, which is in turn the name of getter member
function with the `get` prefix removed, for example to get the value returned
by `M::Config::getModularConfigFolderPath()`:

    $ env_test_cpp --ModularConfigFolderPath

    {
      "ModularConfigFolderPath": "/home/ubuntu/.config/modular"
    }
)");

  ErrorOr<Config> cfg = Config::open();
  if (cfg.isError()) {
    std::cerr << "FAILURE: M::Config::open(): " << cfg.getError() << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "{";
  bool printComma = false;

  auto print = [&](const std::string &key, const std::string &value) {
    if (printComma) {
      std::cout << ",";
    } else {
      printComma = true;
    }
    std::cout << "\n  \"" << key << "\": " << "\"" << value << "\"";
  };

  if (cli.ConfigFilePath) {
    auto configFilePath = cfg->getConfigFilePath();
    if (configFilePath.isError()) {
      std::cerr << "FAILURE: cfg->getConfigFilePath():"
                << configFilePath.getError() << "\n";
      return EXIT_FAILURE;
    }
    print("ConfigFilePath", *configFilePath);
  }

  if (cli.ModularConfigFolderPath) {
    auto configFolderPath = cfg->getModularConfigFolderPath();
    if (configFolderPath.isError()) {
      std::cerr << "FAILURE: cfg->getModularConfigFolderPath(): "
                << configFolderPath.getError() << "\n";
      return EXIT_FAILURE;
    }
    print("ModularConfigFolderPath", *configFolderPath);
  }

  if (cli.ModularDataFolderPath) {
    auto dataFolderPath = cfg->getModularDataFolderPath();
    if (dataFolderPath.isError()) {
      std::cerr << "FAILURE: cfg->getModularConfigFolderPath(): "
                << dataFolderPath.getError() << "\n";
      return EXIT_FAILURE;
    }
    print("ModularDataFolderPath", *dataFolderPath);
  }

  std::cout << "\n}\n";

  return EXIT_SUCCESS;
}
