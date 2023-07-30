//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;

ErrorOr<Config> Config::open() {
  std::filesystem::path homeDirPath = getModularHomeDirPath();

  std::error_code ec;
  // If the modular home directory doesn't even exist, try to create it first.
  if (!std::filesystem::exists(homeDirPath, ec) && !ec)
    std::filesystem::create_directories(homeDirPath, ec);
  if (ec)
    return Error(ec.message());

  // OK great - we have the modular home path, now try and get the config file
  // path.
  std::filesystem::path configFilePath = getConfigFilePath();
  // If we don't have a config, then that's not an error! Simply return an empty
  // config.
  if (!std::filesystem::exists(configFilePath, ec)) {
    if (ec)
      return Error(ec.message());

    return Config{};
  }

  // Set up variables we'll need to get this read.
  Config cfg;
  llvm::SourceMgr sourceMgr;
  unsigned bufferIdx = 0;

  // Check the permissions for the home directory - if it's not writeable then
  // we don't need a lock.
  if (llvm::sys::fs::access(homeDirPath.string(),
                            llvm::sys::fs::AccessMode::Write)) {
    // We don't have write permission here, so we can just read it without a
    // lock.
    auto mBufOr =
        llvm::MemoryBuffer::getFile(configFilePath.string(), /*IsText=*/true);
    if (!mBufOr)
      return Error(mBufOr.getError().message());

    bufferIdx = sourceMgr.AddNewSourceBuffer(std::move(*mBufOr), llvm::SMLoc());
  } else {
    std::optional<Error> error = std::nullopt;
    // Read the file atomically - we may have multiple processes writing.
    ErrorOrSuccess err = readFileUnderLock(
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
  }

  // Grab the memory buffer and parse from it.
  const llvm::MemoryBuffer *mbuf = sourceMgr.getMemoryBuffer(bufferIdx);
  if (ErrorOrSuccess err = cfg.parseFrom(mbuf->getBuffer(), &sourceMgr))
    return err.takeError();

  // Return the initialized configuration.
  return std::move(cfg);
}

ErrorOrSuccess Config::parseFrom(StringRef buffer, llvm::SourceMgr *mgr) {
  auto emitError = [&](llvm::SMLoc loc, Twine msg) -> Error {
    if (!mgr)
      return {msg};

    std::string errMsg;
    llvm::raw_string_ostream stream(errMsg);
    mgr->PrintMessage(stream, loc, llvm::SourceMgr::DK_Error, msg);

    return {errMsg};
  };

  auto takeLine = [&buffer]() -> StringRef {
    size_t newlineLoc = buffer.find_first_of("\n\r\f\v");
    size_t toDrop;
    if (newlineLoc == StringRef::npos) {
      newlineLoc = buffer.size();
      toDrop = newlineLoc;
    } else {
      toDrop = newlineLoc + 1;
    }
    auto line = buffer.take_front(newlineLoc);
    buffer = buffer.drop_front(toDrop);
    return line;
  };

  // While the current pointer is inside the buffer, parse.
  std::string currentSection;
  while (!buffer.empty()) {
    StringRef tmp = takeLine();

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

void Config::setValue(StringRef key, StringRef value) {
  kv[key.lower()] = value;
}

void Config::getValuesInSection(
    StringRef section,
    SmallVectorImpl<std::pair<StringRef, StringRef>> &values) {
  // Iterate all the properties in the map.
  for (auto &properties : kv) {
    // Split on the last '.' - that's the section.
    auto [header, prop] = properties.first().rsplit('.');
    // If the property is empty, that means we didn't have a header (split
    // always fills the first return value). Swap header and prop here cause if
    // we want everything that doesn't have a section, then section should be an
    // empty string.
    if (prop.empty())
      std::swap(header, prop);

    if (header == section)
      values.emplace_back(prop, properties.second);
  }

  // Sort the values so they come out in a deterministic order.
  llvm::stable_sort(values, [](const std::pair<StringRef, StringRef> &lhs,
                               const std::pair<StringRef, StringRef> &rhs) {
    return lhs.first < rhs.first;
  });
}

void Config::flush(raw_ostream &os) {
  std::vector<std::pair<StringRef, std::vector<std::string>>> sections;
  DenseMap<StringRef, unsigned> sectionNameToID;
  for (auto &kV : kv) {
    // Apparently MSVC can't handle structured bindings for some reason.
    StringRef k = kV.first();
    std::string &v = kV.second;

    if (v.empty())
      continue;

    auto [section, prop] = k.rsplit('.');
    if (prop.empty())
      std::swap(section, prop);
    if (section.empty())
      section = "globals";

    auto it = sectionNameToID.try_emplace(section, sections.size());
    if (it.second)
      sections.push_back({section, {}});

    sections[it.first->second].second.push_back((prop + " = " + v).str());
  }

  // Sort the sections to make the output deterministic.
  llvm::stable_sort(sections, [](const auto &lhs, const auto &rhs) {
    // Globals must always come first.
    if (lhs.first == "globals")
      return true;

    return lhs.first < rhs.first;
  });
  for (auto &sectionAndProps : sections)
    llvm::stable_sort(sectionAndProps.second);

  for (auto &sectionAndProps : sections) {
    // Apparently MSVC can't handle structured bindings for some reason.
    StringRef section = sectionAndProps.first;
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
  auto pathOr = writeFileUnderLock(configFilePath,
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

ErrorOrSuccess Config::copyFrom(const Config &other) {
  const llvm::StringMap<std::string> &otherContents = other.getAllValues();
  for (const auto &mapEntry : otherContents) {
    if (kv.contains(mapEntry.first()))
      return Error(Twine("key ") + mapEntry.first() +
                   " already exists in the map");
    kv.insert({mapEntry.first(), mapEntry.second});
  }
  return success();
}
