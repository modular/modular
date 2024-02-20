//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Settings/Settings.h"
#include "Support/Configuration.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/Entitlements/EntitlementStore.h"
#include "Support/ErrorOr.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Settings Implementation
//===----------------------------------------------------------------------===//

Settings::Settings(Config &&cfg, EntitlementStore &&store)
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

static ErrorOr<EntitlementStore> openEntitlementStore(HTTPContextRef httpCtx,
                                                      Config &config) {
  // First, we attempt to open the certificate.
  auto storeOr = EntitlementStore::open(config, httpCtx.copy());
  if (storeOr.isError())
    return storeOr.takeError();
  if (storeOr->has_value())
    return std::move(storeOr->value());

  // This should only open an existing store.
  return Error("unable to open store: try `modular auth`?");
}

ErrorOr<Settings> Settings::open(HTTPContextRef httpCtx, bool createIfMissing,
                                 RefreshPolicy policy) {
  // Open the config.
  auto cfgOr = Config::open();
  if (cfgOr.isError())
    return cfgOr.takeError();

  // Open the entitlement store.
  auto storeOr = openEntitlementStore(httpCtx.copy(), *cfgOr);
  if (!storeOr.isError()) {
    Settings s(cfgOr.takeValue(), std::move(*storeOr));

    // Refresh the certificate if it is necessary to do so.
    if (auto err = s.refresh(std::move(httpCtx), policy))
      return err.takeError();

    return std::move(s);
  }

  // If we have decided that we should not create a new one if it's missing,
  // then simply return an empty one. In the future, this may instead propagate
  // the error above, as entitlements are not available.
  if (!createIfMissing)
    return Settings(cfgOr.takeValue(),
                    EntitlementStore::alwaysOpen(httpCtx.copy(), llvm::errs()));

  // Finally, we don't have one, and we've decided we must have one - generate
  // it.
  auto genOr = EntitlementStore::generate(*cfgOr, std::move(httpCtx));
  if (genOr.isError())
    return genOr.takeError();

  return Settings(cfgOr.takeValue(), std::move(*genOr));
}

ErrorOrSuccess
Settings::refresh(HTTPContextRef httpCtx,
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

const Setting *Settings::get(StringRef key) {
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

bool Settings::set(StringRef key, StringRef value) {
  // Set the same value to appear both internally and within the
  // configuration. First we must assert that either: a) the value doesn't
  // exist locally, or b) is it a configuration value, not an entitlement.
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

ErrorOrSuccess Settings::flush() { return config.flush(); }

const ErrorOr<StringRef> Settings::userID() const {
  return entitlementStore.getUserID();
}

const std::filesystem::path &Settings::clientKeyPriv() const {
  return entitlementStore.clientKeyPrivPath;
}

const std::filesystem::path &Settings::clientKeyPub() const {
  return entitlementStore.clientKeyPubPath;
}

const std::filesystem::path &Settings::clientCert() const {
  return entitlementStore.clientCertPath;
}

const std::filesystem::path &Settings::CRL() const {
  return entitlementStore.crlPath;
}
