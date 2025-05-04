//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;

/// Folder type for searching.
namespace {
enum class FolderType { Config, Data, Cache };
} // namespace

static ErrorOrSuccess createPath(const std::filesystem::path &path) {
  std::error_code ec;
  if (!create_directories(path, ec)) {
    // The directory did not exist, and we failed to create it.
    return Error(Twine(path.string()) +
                 " could not be created: " + ec.message());
  }
  return success();
}

ErrorOr<Config> Config::open() {
  auto configFilePathOr = getConfigFilePath(/*create=*/false);

  // If we don't have a config, then that's not an error! Simply return an empty
  // config. An error is returned above only if the directory cannot be created.
  if (configFilePathOr.isError())
    return Config();
  std::error_code ec;
  if (!std::filesystem::exists(*configFilePathOr, ec)) {
    if (ec)
      return Error(ec.message());
    return Config();
  }

  // Set up variables we'll need to get this read.
  Config cfg;
  llvm::SourceMgr sourceMgr;
  unsigned bufferIdx = 0;

  // Check the permissions for the directory containing the configuration. If
  // it's not writeable, then we avoid acquiring the lock.
  if (llvm::sys::fs::access(configFilePathOr->parent_path().string(),
                            llvm::sys::fs::AccessMode::Write)) {
    // We don't have write permission here, so we can just read it without a
    // lock.
    auto mBufOr = llvm::MemoryBuffer::getFile(configFilePathOr->string(),
                                              /*IsText=*/true);
    if (!mBufOr)
      return Error(mBufOr.getError().message());

    bufferIdx = sourceMgr.AddNewSourceBuffer(std::move(*mBufOr), llvm::SMLoc());
  } else {
    std::optional<Error> error = std::nullopt;
    // Read the file atomically - we may have multiple processes writing.
    ErrorOrSuccess err = readFileUnderLock(
        *configFilePathOr, [&](const std::filesystem::path &filePath) {
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
  auto emitError = [&](llvm::SMLoc loc, const Twine &msg) -> Error {
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
      upper.begin(), upper.end(),
      [](char c) { return (c == '.') || (c == '-'); }, '_');

  if (allowEnvOverride) {
    // Check for this environment variable.
    auto envOr = llvm::sys::Process::GetEnv("MODULAR_" + upper);
    // If we have this env variable, save it in the map. We don't care if it
    // overrides something.
    if (envOr)
      kv[key.lower()] = *envOr;
  }

  return kv[key.lower()];
}

StringRef Config::getValueOr(llvm::StringRef key,
                             llvm::StringRef defaultValue) {
  StringRef stringValue = getValue(key);
  if (stringValue.empty())
    return defaultValue;
  return stringValue;
}

bool Config::getValueAsBool(StringRef key, bool defaultValue) {
  auto stringValue = getValue(key);
  if (stringValue.empty())
    return defaultValue;
  return llvm::StringSwitch<bool>(stringValue)
      .CasesLower("0", "false", "no", false)
      .CasesLower("1", "true", "yes", true)
      .Default(defaultValue);
}

void Config::setValue(StringRef key, StringRef value) {
  kv[key.lower()] = value;
}

void Config::populateEnvOverrides() {
  if (allowEnvOverride)
    for (const auto &[k, _] : kv)
      getValue(k);
}

/// Get the list of search paths, in order of preference.
static void getSearchPaths(SmallVectorImpl<std::filesystem::path> &paths,
                           FolderType type) {
  // If MODULAR_HOME is defined, use that and only that.
  auto modularHome = llvm::sys::Process::GetEnv("MODULAR_HOME");
  if (modularHome) {
    // Cache folder is a subdirectory in this case.
    if (type == FolderType::Cache) {
      paths.push_back(std::filesystem::path(*modularHome) / "cache");
      return;
    }
    paths.push_back(*modularHome);
    return;
  }

  // If MODULAR_DERIVED_PATH is defined, use that and only that.
  auto derivedPath = llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH");
  if (derivedPath) {
    // Cache folder is a subdirectory in this case.
    if (type == FolderType::Cache) {
      paths.push_back(std::filesystem::path(*derivedPath) / "cache");
      return;
    }
    paths.push_back(*derivedPath);
    return;
  }

  // To work well in test environments, check for a standardized test
  // environment variable. This is always the last option, if available.
  auto testTempdir = llvm::sys::Process::GetEnv("TEST_TMPDIR");
  if (testTempdir) {
    paths.push_back(std::filesystem::path(*testTempdir) / ".modular");
    return;
  }

#ifndef _WIN32
  // To support existing installs, add $HOME/.modular if it exists. If it
  // does it always takes precedence.
  auto homeDir = llvm::sys::Process::GetEnv("HOME");
  bool addedHome = false;
  if (homeDir) {
    auto path = std::filesystem::path(*homeDir) / ".modular";
    if (std::filesystem::exists(path)) {
      if (type == FolderType::Cache)
        paths.push_back(path / ".cache" / "modular");
      else
        paths.push_back(std::filesystem::path(*homeDir) / ".modular");
      paths.push_back(path);
      addedHome = true;
    }
  }

  // Second we check for XDG_CONFIG_HOME and XDG_DATA_HOME. If they exist, we
  // use them. We deviate from the spec here and use $HOME/.config/modular and
  // `$HOME/.local/share/modular` as the spec says when the XDG_*_HOME variables
  // are not set. We will instead still default to `$HOME/.modular`.
  //
  // BOTH of these must be set for us to use them otherwise things may break.
  ///
  // XDG_CACHE_HOME is optional and if it is not set we will use
  // $HOME/.cache/modular if we are otherwise using XDG_CONFIG_HOME and
  // XDG_DATA_HOME too.
  auto xdgConfigHome = llvm::sys::Process::GetEnv("XDG_CONFIG_HOME");
  auto xdgConfigData = llvm::sys::Process::GetEnv("XDG_DATA_HOME");
  auto xdgConfigCache = llvm::sys::Process::GetEnv("XDG_CACHE_HOME");
  if (xdgConfigHome && xdgConfigData) {
    if (!xdgConfigCache && homeDir)
      xdgConfigCache = std::filesystem::path(*homeDir) / ".cache" / "modular";
    switch (type) {
    case FolderType::Config:
      paths.push_back(std::filesystem::path(*xdgConfigHome) / "modular");
      break;
    case FolderType::Data:
      paths.push_back(std::filesystem::path(*xdgConfigData) / "modular");
      break;
    case FolderType::Cache:
      if (!xdgConfigCache)
        paths.push_back(std::filesystem::path(*xdgConfigCache) / "modular");
      break;
    }
  }

  // Lastly if we haven't added $HOME/.modular in first step, add it now.
  if (!addedHome && homeDir) {
    if (type == FolderType::Cache)
      paths.push_back(std::filesystem::path(*homeDir) / ".modular" / "cache");
    else
      paths.push_back(std::filesystem::path(*homeDir) / ".modular");
  }

  // Add /opt/modular as a global destination.
  paths.push_back("/opt/modular");
#else  // _WIN32
  // Add $APPDATA\Local\Modular
  auto defaultRoot = llvm::sys::Process::GetEnv("APPDATA");
  assert(defaultRoot.has_value() && "Must have APPDATA");
  paths.push_back(std::filesystem::path(*defaultRoot) / "Local" / "Modular");
#endif // _WIN32
}

static ErrorOr<std::filesystem::path> findBestPathForType(FolderType type,
                                                          bool create) {
  // Get the list of search paths.
  SmallVector<std::filesystem::path, 3> searchPaths;
  getSearchPaths(searchPaths, type);

  // Check each of the search paths for existence - if none of them exist
  // return the first of the paths as MODULAR_HOME.
  auto found =
      llvm::find_if(searchPaths, [&](const std::filesystem::path &path) {
        std::error_code ec;
        bool exists = std::filesystem::exists(path, ec);
        assert(!ec && "error checking for path existence");
        return exists;
      });
  if (found != searchPaths.end())
    return *found;

  // If we aren't supposed to create the directory, then just return the path
  // directly. It is still our "best choice", even if we can't use it. The
  // caller must specifically set create=false to exercise this path.
  if (!create)
    return searchPaths[0];

  // None of the above directories exist. Attempt to create a directory, in the
  // order provided. We iterate and return the first one we can create.
  Error firstErr = Error("no candidates for directory");
  bool firstErrFound = false;
  found = llvm::find_if(searchPaths, [&](const std::filesystem::path &path) {
    auto err = createPath(path);
    if (err.isError()) {
      if (!firstErrFound) {
        firstErrFound = true;
        firstErr = err.takeError();
      }
      return false;
    }
    return true;
  });
  if (found != searchPaths.end())
    return *found;

  // Nothing could be created. Return the first error encountered (which is the
  // directory we'd want to use with the highest priority).
  return firstErr;
}

ErrorOr<std::filesystem::path> Config::getModularConfigFolderPath(bool create) {
  return findBestPathForType(FolderType::Config, create);
}

ErrorOr<std::filesystem::path> Config::getModularDataFolderPath(bool create) {
  return findBestPathForType(FolderType::Data, create);
}

ErrorOr<std::filesystem::path> Config::getConfigFilePath(bool create) {
  constexpr llvm::StringLiteral kModularConfigFileName = "modular.cfg";

  // If we found the config file this way, then return it.
  auto configFile = findModularFile(kModularConfigFileName);
  if (configFile)
    return *configFile;

  // Otherwise, return where it should be placed.
  auto configFolderOr = getModularConfigFolderPath(create);
  if (configFolderOr.isError())
    return configFolderOr.takeError();
  return *configFolderOr / kModularConfigFileName.str();
}

void Config::setEnvOverride(bool newVal) { allowEnvOverride = newVal; }

std::optional<std::filesystem::path> M::findModularFile(StringRef fileName) {
  SmallVector<std::filesystem::path, 4> searchPaths;
  getSearchPaths(searchPaths, FolderType::Config);
#ifndef _WIN32
  // Append a path to the search paths on UNIX systems.
  searchPaths.push_back(std::filesystem::path("/etc/modular"));
#endif // _WIN32

#ifdef __APPLE__
  // Homebrew installs into /opt/homebrew on arm64 and we symlink our /etc
  // package files into HOMEBREW_PREFIX/etc/modular location.
  searchPaths.push_back(std::filesystem::path("/opt/homebrew/etc/modular"));
#endif // __APPLE__

  // Try to find the file in the provided paths.
  auto found =
      llvm::find_if(searchPaths, [&](const std::filesystem::path &path) {
        std::error_code ec;
        bool exists = std::filesystem::exists(path / fileName.str(), ec);
        assert(!ec && "error checking for path existence");
        return exists;
      });

  // Was not found, return nullopt.
  if (found == searchPaths.end())
    return std::nullopt;

  // We did find it, return that path.
  return *found / fileName.str();
}
