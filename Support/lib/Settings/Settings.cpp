//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Settings/Settings.h"
#include "Support/Configuration.h"
#include "Support/Context.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/Entitlements/EntitlementStore.h"
#include "Support/ErrorOr.h"

//===----------------------------------------------------------------------===//
// Settings::Impl
//===----------------------------------------------------------------------===//

namespace M {
/// This class defines the implementation for the Settings object. We use the
/// pImpl idiom here because we want to hide the implementation details (Config,
/// EntitlementStore) from the user as much as possible.
class Settings::Impl {
public:
  /// Construct a new Impl. The Impl object will take ownership of `cfg` and
  /// `store`.
  Impl(Config &&cfg, EntitlementStore &&store);

  /// Create a new Impl object with an HTTPContextRef. If `createIfMissing` is
  /// `true`, not having a certificate on the system already means that we will
  /// simply call `generate` immediately afterwards.
  static ErrorOr<std::unique_ptr<Impl>> create(HTTPContextRef httpCtx,
                                               bool createIfMissing);

  /// Get a pointer to a setting identified by `key`.
  const Setting *get(StringRef key);

  /// Set the value indicated by `key`. Returns true if successful, false
  /// if this value is controlled by an entitlement.
  bool set(StringRef key, StringRef value);

  /// Flush the full configuration.
  ErrorOrSuccess flush();

  /// Refresh the settings - this may invalidate any pointers returned by the
  /// `get` API.
  ErrorOrSuccess refresh(HTTPContextRef httpCtx,
                         RefreshPolicy shouldRefreshEntitlements);

private:
  Config config;
  EntitlementStore entitlementStore;
  llvm::StringMap<Setting> settings;

  // Allow the Settings class to reach in and access the entitlementStore
  // directly, if only for making the various certificate paths available in
  // other contexts.
  friend class Settings;
};
} // namespace M

using namespace M;

//===----------------------------------------------------------------------===//
// Settings::Impl Implementation
//===----------------------------------------------------------------------===//

Settings::Impl::Impl(Config &&cfg, EntitlementStore &&store)
    : config(std::move(cfg)), entitlementStore(std::move(store)) {
  // Set the user ID first up. This should come from the entitlement store,
  // not from the config.
  auto userIDOr = entitlementStore.getUserID();
  // TODO(#27787): This should become a hard error if the entitlement store
  //               doesn't have a value once certificates are rolled-out.
  if (!userIDOr.isError())
    settings.try_emplace("user.id", Setting{*userIDOr});

  // Populate any env overrides we might have.
  config.populateEnvOverrides();

  // Read all the values in the Config right now. This will read any env
  // variables that might be set and save those values immediately. These have
  // to be read greedily, but the entitlements we can populate lazily as
  // they're requested.
  for (const auto &[k, v] : config.getAllValues())
    settings.try_emplace(k, Setting{v});
}

ErrorOr<std::unique_ptr<Settings::Impl>>
Settings::Impl::create(HTTPContextRef httpCtx, bool createIfMissing) {
  // Open the config.
  auto cfgOr = Config::open();
  if (cfgOr.isError())
    return cfgOr.takeError();

  // First, we attempt to open the certificate.
  auto storeOr = EntitlementStore::open(*cfgOr, httpCtx.copy());
  if (storeOr.isError())
    return storeOr.takeError();

  // We have one, so we should use it.
  if (storeOr->has_value())
    return std::make_unique<Impl>(cfgOr.takeValue(), *storeOr.takeValue());

  // If we have decided that we should not create a new one if it's missing,
  // then simply return an empty one.
  if (!createIfMissing)
    return std::make_unique<Impl>(
        cfgOr.takeValue(),
        EntitlementStore::alwaysOpen(httpCtx.copy(), llvm::errs()));

  // Finally, we don't have one, and we've decided we must have one - generate
  // it.
  auto genOr = EntitlementStore::generate(*cfgOr, std::move(httpCtx));
  if (genOr.isError())
    return genOr.takeError();

  return std::make_unique<Impl>(cfgOr.takeValue(), genOr.takeValue());
}

bool Settings::Impl::set(StringRef key, StringRef value) {
  // Set the same value to appear both internally and within the configuration.
  // First we must assert that either: a) the value doesn't exist locally, or
  // b) is it a configuration value, not an entitlement.
  auto found = settings.find(key);
  if (found != settings.end()) {
    if (!llvm::isa_and_present<StringRef>(&found->second))
      return false; // Is not a configuration value.
  }

  // Replace or insert this false.
  config.setValue(key, value);
  settings.try_emplace(key, Setting{value});
  return true;
}

ErrorOrSuccess Settings::Impl::flush() { return config.flush(); }

const Setting *Settings::Impl::get(StringRef key) {
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

ErrorOrSuccess
Settings::Impl::refresh(HTTPContextRef httpCtx,
                        Settings::RefreshPolicy shouldRefreshEntitlements) {
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
  return entitlementStore.refreshIfNecessary(std::move(httpCtx),
                                             shouldRefreshEntitlements);
}

//===----------------------------------------------------------------------===//
// Settings Implementation
//===----------------------------------------------------------------------===//

Settings::~Settings() = default;

ErrorOr<Settings> Settings::open(HTTPContextRef httpCtx, bool createIfMissing,
                                 RefreshPolicy policy) {
  auto implOr = Impl::create(httpCtx.copy(), createIfMissing);
  if (implOr.isError())
    return implOr.takeError();

  // Create the implementation.
  Settings s;
  s.impl = std::move(*implOr);

  // Refresh the certificate if it is necessary to do so. We only care about an
  // error here if createIfMissing is specified. Otherwise, we move on.
  if (auto err = s.impl->refresh(std::move(httpCtx), policy))
    if (createIfMissing)
      return err.takeError();

  return std::move(s);
}

const Setting *Settings::get(StringRef key) const { return impl->get(key); }

bool Settings::set(StringRef key, StringRef value) {
  return impl->set(key, value);
}

ErrorOrSuccess Settings::flush() { return impl->flush(); }

const std::filesystem::path &Settings::clientKeyPriv() const {
  return impl->entitlementStore.clientKeyPrivPath;
}

const std::filesystem::path &Settings::clientKeyPub() const {
  return impl->entitlementStore.clientKeyPubPath;
}

const std::filesystem::path &Settings::clientCert() const {
  return impl->entitlementStore.clientCertPath;
}

const std::filesystem::path &Settings::CRL() const {
  return impl->entitlementStore.crlPath;
}
