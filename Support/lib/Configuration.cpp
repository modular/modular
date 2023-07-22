//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;

ErrorOr<Config> Config::open() {
  std::filesystem::path configFilePath = getConfigFilePath();

  // If we don't have a config, then that's not an error! Simply return an empty
  // config.
  std::error_code ec;
  if (!std::filesystem::exists(configFilePath, ec)) {
    if (ec)
      return Error(ec.message());

    return Config{};
  }

  // Set up variables we'll need to get this read.
  Config cfg;
  llvm::SourceMgr sourceMgr;
  std::optional<Error> error = std::nullopt;
  unsigned bufferIdx = 0;
  // Read the file atomically - we may have multiple processes writing.
  ErrorOrSuccess err = readFileAtomically(
      configFilePath, [&](const std::filesystem::path &filePath) {
        auto mBufOr =
            llvm::MemoryBuffer::getFile(filePath.string(), /*IsText=*/true);
        if (!mBufOr) {
          error = Error(mBufOr.getError().message());
          return;
        }

        bufferIdx =
            sourceMgr.AddNewSourceBuffer(std::move(*mBufOr), llvm::SMLoc());
      });
  // Check for errors.
  if (err.isError())
    return err.takeError();
  if (error.has_value())
    return std::move(*error);

  // Grab the memory buffer and parse from it.
  const llvm::MemoryBuffer *mbuf = sourceMgr.getMemoryBuffer(bufferIdx);
  if ((err = cfg.parseFrom(mbuf->getBuffer(), &sourceMgr)))
    return err.takeError();

  // Return the initialized configuration.
  return std::move(cfg);
}

ErrorOrSuccess Config::parseFrom(StringRef buffer, llvm::SourceMgr *mgr) {
  const char *curPtr = buffer.begin();

  auto emitError = [&](llvm::SMLoc loc, Twine msg) -> Error {
    if (!mgr)
      return {msg};

    std::string errMsg;
    llvm::raw_string_ostream stream(errMsg);
    mgr->PrintMessage(stream, loc, llvm::SourceMgr::DK_Error, msg);

    return {errMsg};
  };

  auto takeLine = [&curPtr](size_t &outsz) {
    while (true) {
      char c = *curPtr++;
      switch (c) {
      case '\n':
      case '\r':
      case '\f':
      case '\v':
        return;
      default:
        ++outsz;
        continue;
      }
    }
  };

  // While the current pointer is inside the buffer, parse.
  std::string currentSection;
  size_t lineLen = 0;
  while (curPtr < buffer.end()) {
    auto resetLineLen = llvm::make_scope_exit([&]() { lineLen = 0; });

    llvm::SMLoc lineStart = llvm::SMLoc::getFromPointer(curPtr);
    takeLine(lineLen);
    // Build a StringRef from this.
    StringRef tmp(lineStart.getPointer(), lineLen);

    // If there's nothing but whitespace in it, continue.
    if (tmp.trim().empty())
      continue;

    // Parse a section delimiter.
    if (tmp.consume_front("[")) {
      tmp = tmp.take_until([](char c) { return c == ']'; }).trim();
      currentSection = tmp;
      continue;
    }

    // If there's a comment at the end of this line, drop it and parse
    // anything in front of it.
    tmp = tmp.take_until([](char c) { return c == '#' || c == ';'; });

    // Again, if it's empty, drop this line.
    if (tmp.trim().empty())
      continue;

    // Split on the equals sign.
    auto [k, v] = tmp.split('=');
    if (v.empty()) {
      return emitError(llvm::SMLoc::getFromPointer(k.begin()),
                       "malformed line: expected `key = value`");
    }

    // Allow global properties - anything not in a section. The way this works
    // is by not prefixing with the current section.
    std::string currentSectionPrefix =
        (!currentSection.empty() ? (currentSection + ".") : "");

    // Insert the key and value into the current map, trimming off any extra
    // whitespace. We insert each value under the key `section.key` to make
    // lookups fast.
    std::string constructedKey = (currentSectionPrefix + k.trim()).str();

    // Insert under the lowercase name - section names and property names
    // are case-insensitive. This will also take a copy of the value string
    // so that we don't have to deal with keeping the buffers alive.
    kv[StringRef(constructedKey).lower()] = v.trim();
  }
  return success();
}

StringRef Config::getValue(StringRef key) {
  std::string upper = key.upper();
  std::replace_if(
      upper.begin(), upper.end(), [](char c) { return c == '.'; }, '_');

  // Check for this environment variable.
  auto envOr = llvm::sys::Process::GetEnv("MODULAR_" + upper);
  // If we have this env variable, save it in the map. We don't care if it
  // overrides something.
  if (envOr)
    kv[key.lower()] = *envOr;

  return kv[key.lower()];
}

void Config::flush(raw_ostream &os) {
  llvm::StringMap<std::vector<std::string>> map;

  for (auto &kV : kv) {
    // Apparently MSVC can't handle structured bindings for some reason.
    StringRef k = kV.first();
    std::string &v = kV.second;

    auto [section, prop] = k.rsplit('.');
    if (prop.empty())
      std::swap(section, prop);

    std::vector<std::string> *props = nullptr;
    if (!section.empty())
      props = &map[section];
    if (!props)
      props = &map["globals"];

    assert(props && "every property must fit in at least one category");
    props->push_back((prop + " = " + v).str());
  }

  for (auto &sectionAndProps : map) {
    // Apparently MSVC can't handle structured bindings for some reason.
    StringRef section = sectionAndProps.first();
    std::vector<std::string> &props = sectionAndProps.second;

    if (section != "globals")
      os << "[" << section << "]";
    os << "\n";

    // First, sort the properties.
    llvm::stable_sort(props);
    for (auto &prop : props)
      os << prop << "\n";
    os << "\n";
  }
}

ErrorOrSuccess Config::flush() {
  std::filesystem::path configFilePath = getConfigFilePath();

  // Write the config file to the output atomically.
  auto pathOr = writeFileAtomically(configFilePath,
                                    [&](llvm::raw_ostream &os) { flush(os); });
  if (pathOr.isError())
    return pathOr.takeError();

  return success();
}

std::filesystem::path Config::getModularHomeDirPath() {
  auto modularHome = llvm::sys::Process::GetEnv("MODULAR_HOME");
  // No env variable, find it in PATH.
  if (!modularHome) {
#ifdef _WIN32
    modularHome = findDirInEnvPath(".modular", "PATH", ';');
#else
    modularHome = findDirInEnvPath(".modular");
#endif
  }

  // If we have MODULAR_DERIVED_PATH, treat that as MODULAR_HOME.
  if (!modularHome)
    modularHome = llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH");

  // Default to CWD - no env variable and nothing in PATH, so just use CWD.
  if (!modularHome)
    return ".modular";

  return *modularHome;
}

std::filesystem::path Config::getConfigFilePath() {
  constexpr llvm::StringLiteral kModularConfigFileName = "modular.cfg";
  return getModularHomeDirPath() / kModularConfigFileName.str();
}
