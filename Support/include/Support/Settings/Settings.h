//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SETTINGS_SETTINGS_H
#define SUPPORT_SETTINGS_SETTINGS_H

#include "Support/Configuration.h"
#include "Support/Entitlements/EntitlementStore.h"
#include "Support/ErrorOr.h"
#include "Support/HTTP/HTTPClient.h"
#include "Support/Settings/Setting.h"

#include <filesystem>
#include <mutex>

namespace M {
/// This class provides the collection of system settings. Users should use the
/// `open` method to get the current settings, and should use `refresh` when
/// it's necessary to do so. Refreshing the settings means intentionally looking
/// for new environment variables, in the case of the config, and fetching a new
/// certificate if necessary, in the case of the entitlement store. This
/// semi-inextricably ties the notion of "refreshing the settings" to the notion
/// of "refreshing authentication", which is not really desirable, but it's
/// difficult to extricate these two concepts in a way that makes sense since
/// the carrier for both signals is exactly the same.
class Settings {
public:
  /// Define the various constructors - this object must be default
  /// constructible, it must be non-copyable, and it must be move-able.
  ~Settings() = default;
  Settings(const Settings &other) = delete;
  Settings(Settings &&other) = default;

  /// Create new settings. Note that this should be used for tests only,
  /// all regular code paths should use Settings::open.
  Settings(Config &&cfg, EntitlementStore &&store);

  /// Get a setting by its key. This corresponds to Config::getValue.
  const Setting *get(StringRef key);

  /// Get a setting of type T by its key. Asserts that the setting is of type T.
  template <typename T>
  T get(StringRef key) {
    return llvm::cast_if_present<T>(get(key));
  }

  /// As a special case, support a get that cast to bool using a sensible
  /// and standardize way of evaluation. If the entitlement exists, or if
  /// it is a StringRef with an appropriate value then true is returned.
  bool getBool(StringRef key, const bool defaultValue) {
    const Setting *s = get(key);
    if (s == nullptr)
      return defaultValue;
    auto stringValue = llvm::cast<StringRef>(s);
    int intResult = llvm::StringSwitch<int>(stringValue)
                        .CasesLower("0", "false", "no", 0)
                        .CasesLower("1", "true", "yes", 1)
                        .Default(-1);
    if (intResult == -1)
      return false;
    return bool(intResult);
  }

  /// Sets a value in the underlying config only. The caller should be careful
  /// to check the return value; this will fail if the value is not settable.
  bool set(StringRef key, StringRef value);

  /// Clear deletes a value from the underlying config only. See `set`.
  bool clear(StringRef key);

  /// Flushes the configuration. An error will be raised here if a set failed
  /// previously (and e.g. conflicted with an entitlement).
  ErrorOrSuccess flush();

  /// Get an entitlement of the explicit type.
  template <class T>
  bool getBool() {
    // For entitlements, this evaluates only as a simple existence test using
    // the get method above. Note that getBool() on an entitlement name will
    // not return true, the actual entitlement must be present.
    auto ptr = std::make_unique<T>();
    const Setting *s = get(ptr->getName());
    return s != nullptr;
  }

  /// Return the current userID.
  ErrorOr<StringRef> userID() const;

  /// PEM-encoded client private key (not a path).
  const std::string clientKeyPriv();

  /// PEM-encoded client certificate (not a path).
  const std::string clientCert();

  /// Refresh the settings if it's necessary to do so. This will refresh all
  /// configurations and the entitlement store. The user can configure a policy
  /// on when a refresh is 'necessary', using the validFrom and validTo values
  /// of the certificate, converted to system clock time points. If no policy is
  /// provided, a default of 'halfway between from and to' is used.
  using RefreshPolicy =
      llvm::function_ref<bool(std::chrono::system_clock::time_point from,
                              std::chrono::system_clock::time_point to)>;

  enum EntitlementPolicy {
    /// Always succeed regardless of underlying entitlements.
    kAlwaysSucceed = 0,

    /// Fail if no entitlement store is available (returning an error).
    kRequiredNoPrompt = 1,

    /// Prompt for authentication, prompting if this fails. Will open web
    /// browser with auth link.
    kRequiredWithPrompt = 2,

    /// Prompt for authentication without opening a browser window.
    kRequiredWithPromptNoBrowser = 3,
  };

  /// Open the current configuration and entitlement store, refreshing if
  /// needed, and return a Settings object.
  ///
  /// This function will open and parse the local configuration, and may
  /// refresh the local entitlement store. It should only be called once
  /// at start-up, and then stored in a context for future reference.
  static ErrorOr<Settings> open(HTTPContextRef httpCtx,
                                EntitlementPolicy entitlements = kAlwaysSucceed,
                                RefreshPolicy policy = nullptr);

private:
  /// Used internally on creation.
  ErrorOrSuccess refresh(HTTPContextRef httpCtx,
                         RefreshPolicy shouldRefreshEntitlements);

  Config config;
  EntitlementStore entitlementStore;
  struct impl {
    std::mutex mu;
    llvm::StringMap<Setting> map;
  };
  std::unique_ptr<impl> settings;
};
} // namespace M

#endif // SUPPORT_SETTINGS_SETTINGS_H
