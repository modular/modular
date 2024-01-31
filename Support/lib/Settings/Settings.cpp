//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Settings/Settings.h"
#include "Support/Configuration.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/Entitlements/EntitlementStore.h"

using namespace M;

namespace M {
class Settings::Impl {
public:
  Impl(Config &&cfg, EntitlementStore &&store)
      : config(std::move(cfg)), entitlementStore(std::move(store)) {
    // Populate any env overrides we might have.
    config.populateEnvOverrides();

    // Read all the values in the Config right now. This will read any env
    // variables that might be set and save those values immediately. These have
    // to be read greedily, but the entitlements we can populate lazily as
    // they're requested.
    for (const auto &[k, v] : config.getAllValues())
      settings.try_emplace(k, Setting{v});
  }

  static ErrorOr<std::unique_ptr<Impl>> create(HTTPClient *client) {
    auto cfgOr = Config::open();
    if (cfgOr.isError())
      return cfgOr.takeError();

    // Attempt to open the entitlement store.
    auto storeOr = EntitlementStore::open(client);
    if (storeOr.isError())
      return storeOr.takeError();

    // No entitlement store, use an empty one. This is the 'fail open' method.
    if (!*storeOr)
      return std::make_unique<Impl>(cfgOr.takeValue(), EntitlementStore{});

    return std::make_unique<Impl>(cfgOr.takeValue(), *storeOr.takeValue());
  }

  const Setting *get(StringRef key) {
    // Try to find the setting in the config map.
    auto found = settings.find(key);
    if (found != settings.end())
      return &found->second;

    // Try to look it up in the entitlement store if we didn't find it already.
    if (auto *e = entitlementStore.getEntitlement(key)) {
      // Cache the entitlement in the settings map.
      auto [iter, _] = settings.try_emplace(key, Setting{e});
      return &iter->second;
    }

    // Didn't find anything, return nullptr.
    return nullptr;
  }

  ErrorOrSuccess refresh(HTTPClient &client,
                         RefreshPolicy shouldRefreshEntitlements) {
    // Clear out all settings - this is crucial because any stored references to
    // entitlements or configs may be invalidated.
    settings.clear();

    // Populate any env overrides we might have.
    config.populateEnvOverrides();

    // Refresh the config values. This will re-read any env variables if
    // necessary.
    for (const auto &[k, v] : config.getAllValues())
      settings.try_emplace(k, Setting{v});

    // Simply refresh the entitlement store - it has all the logic necessary to
    // do that internally.
    return entitlementStore.refreshIfNecessary(client,
                                               shouldRefreshEntitlements);
  }

private:
  Config config;
  EntitlementStore entitlementStore;
  llvm::StringMap<Setting> settings;
};
} // namespace M

ErrorOr<Settings> Settings::open(HTTPClient *client) {
  Settings s;
  auto implOr = Impl::create(client);
  if (implOr.isError())
    return implOr.takeError();

  s.impl = std::move(*implOr);

  return s;
}

const Setting *Settings::get(StringRef key) { return impl->get(key); }

ErrorOrSuccess Settings::refresh(HTTPClient &client,
                                 RefreshPolicy shouldRefreshEntitlements) {
  return impl->refresh(client, shouldRefreshEntitlements);
}
