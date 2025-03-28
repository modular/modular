//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Settings/Settings.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Settings Implementation
//===----------------------------------------------------------------------===//

Settings::Settings(Config &&cfg)
    : config(std::move(cfg)), settings(std::make_unique<impl>()) {
  // Populate any env overrides we might have.
  config.populateEnvOverrides();

  // Read all the values in the Config right now. This will read any env
  // variables that might be set and save those values immediately. These have
  // to be read greedily, but the entitlements we can populate lazily as
  // they're requested.
  for (const auto &[k, v] : config.getAllValues())
    settings->map.try_emplace(k, Setting{v});
}

ErrorOr<Settings> Settings::open() {
  auto cfgOr = Config::open();
  if (cfgOr.isError())
    return cfgOr.takeError();

  return Settings(cfgOr.takeValue());
}

const Setting *Settings::get(StringRef key) {
  std::lock_guard<std::mutex> lk(settings->mu);

  // Try to find the setting in the config map.
  auto found = settings->map.find(key);
  if (found != settings->map.end())
    return &found->second;

  // Finally, this may be in the environment and something that we don't
  // know about already. In that case, we populate the setting as well.
  auto value = config.getValue(key);
  if (!value.empty()) {
    auto [iter, _] = settings->map.try_emplace(key, Setting{value});
    return &iter->second;
  }

  return nullptr;
}

bool Settings::set(StringRef key, StringRef value) {
  std::lock_guard<std::mutex> lk(settings->mu);

  // Set the same value to appear both internally and within the
  // configuration. First we must assert that either: a) the value doesn't
  // exist locally, or b) is it a configuration value, not an entitlement.
  auto found = settings->map.find(key);
  if (found != settings->map.end()) {
    if (!llvm::isa_and_present<StringRef>(&found->second))
      return false; // Is not a configuration value.
  }

  // Replace or insert this false.
  config.setValue(key, value);
  settings->map.try_emplace(key, Setting{value});
  return true;
}

bool Settings::clear(StringRef key) { return set(key, ""); }
