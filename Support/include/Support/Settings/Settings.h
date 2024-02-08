//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SETTINGS_SETTINGS_H
#define SUPPORT_SETTINGS_SETTINGS_H

#include "Support/HTTP/HTTPClient.h"
#include "Support/Settings/Setting.h"

namespace M {
/// Forward-declare the Context object.
class Context;

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
  /// Explicit destructor so that the private impl can be handled correctly.
  ~Settings();

  /// Define the various constructors - this object must be default
  /// constructible, it must be non-copyable, and it must be move-able.
  Settings() = default;
  Settings(const Settings &other) = delete;
  Settings(Settings &&other) = default;

  /// Open the current system settings and emplace them in the context object.
  /// If we don't have secured settings and `createIfMissing` is false, then we
  /// will not fetch the secured settings. To be concrete, there are 4 states
  /// that we could be in:
  ///  (1) No secured settings, don't bother getting them (this should go away).
  ///  (2) No secured settings, get them.
  ///  (3) Have secured settings, use them.
  ///  (4) Have secured settings, fetch them again (this is refresh).
  /// `createIfMissing` selects between states (1) and (2). State (3) is always
  /// the default.
  static ErrorOrSuccess emplace(Context &ctx, HTTPClient *client,
                                bool createIfMissing = false);

  /// Get a setting by its key. This corresponds to Config::getValue.
  const Setting *get(StringRef key) const;

  /// Get a setting of type T by its key. Asserts that the setting is of type T.
  template <typename T>
  T get(StringRef key) const {
    return llvm::cast_if_present<T>(get(key));
  }

  /// Refresh the settings if it's necessary to do so. This will refresh all
  /// configurations and the entitlement store. The user can configure a policy
  /// on when a refresh is 'necessary', using the validFrom and validTo values
  /// of the certificate, converted to system clock time points. If no policy is
  /// provided, a default of 'halfway between from and to' is used.
  using RefreshPolicy =
      llvm::function_ref<bool(std::chrono::system_clock::time_point from,
                              std::chrono::system_clock::time_point to)>;
  ErrorOrSuccess refresh(HTTPClient &client,
                         RefreshPolicy shouldRefreshEntitlements = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace M

#endif // SUPPORT_SETTINGS_SETTINGS_H
