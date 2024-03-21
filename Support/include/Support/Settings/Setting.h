//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SETTINGS_SETTING_H
#define SUPPORT_SETTINGS_SETTING_H

#include "Support/ADT/GenericUniquePtr.h"
#include "Support/ADT/SmartVariant.h"
#include "llvm/Support/Casting.h"

namespace M {
/// Forward-declare the Settings class.
class Settings;

/// Forward-declare M::Entitlement.
class Entitlement;

/// This class provides an abstraction over config and entitlement values.
/// Essentially, our system has protected settings (entitlements) and
/// unprotected settings (config strings). This class provides an abstraction
/// over both.
class Setting {
public:
  /// Get the raw underlying storage. Users should not use this directly, and
  /// should instead use a direct cast of the Setting object like so:
  ///
  ///  const Setting *s = settings.get("modular-developer");
  ///  auto *e = dyn_cast_if_present<ModularDeveloperEntitlement>(setting);
  ///
  /// Because Settings are possibly protected, it's important to confirm the
  /// type of the Setting that you just fetched with llvm::isa_and_present or
  /// llvm::dyn_cast_if_present. If the setting isn't the type you expected, you
  /// should *not* trust it, *especially* if the setting is protected!
  const auto &getStorage() const { return storage; }

private:
  /// It must not be possible to construct a Setting object outside the Settings
  /// class.
  friend class Settings;

  Setting(StringRef s) : storage(s.str()) {}
  Setting(M::Entitlement *e) : storage(e) {}

  SmartVariant<std::string, M::Entitlement *> storage;
};
} // namespace M

namespace llvm {
/// Provide casting support from an M::Setting to a concrete Entitlement, such
/// as ModularDeveloperEntitlement.
template <typename E>
struct CastInfo<E *, M::Setting *,
                std::enable_if_t<std::is_base_of_v<M::Entitlement, E>>> {
  using Self = CastInfo<E *, M::Setting *,
                        std::enable_if_t<std::is_base_of_v<M::Entitlement, E>>>;

  static bool isPossible(const M::Setting *s) {
    if (!isa<M::Entitlement *>(s->getStorage()))
      return false;

    // Cast the storage to a GenericUniquePtr (it's an Entitlement, so we can
    // always cast to an Entitlement *) and use isa.
    return isa<E>(cast<M::Entitlement *>(s->getStorage()));
  }

  static decltype(auto) doCast(const M::Setting *s) {
    return cast<E>(cast<M::Entitlement *>(s->getStorage()));
  }

  static E *castFailed() { return nullptr; }

  static decltype(auto) doCastIfPossible(const M::Setting *s) {
    if (!isPossible(s))
      return Self::castFailed();

    return doCast(s);
  }
};

template <typename E>
struct CastInfo<const E *, const M::Setting *,
                std::enable_if_t<std::is_base_of_v<M::Entitlement, E>>>
    : public ConstStrippingForwardingCast<const E *, const M::Setting *,
                                          CastInfo<E *, M::Setting *>> {};

/// Provide casting support from an M::Setting to a StringRef. This is primarily
/// useful for Config settings which are represented as strings.
template <>
struct CastInfo<StringRef, M::Setting *>
    : public DefaultDoCastIfPossible<StringRef, M::Setting *,
                                     CastInfo<StringRef, M::Setting *>> {
  static bool isPossible(const M::Setting *s) {
    return isa<std::string>(s->getStorage());
  }

  static StringRef doCast(const M::Setting *s) {
    return llvm::cast<std::string>(s->getStorage());
  }

  static StringRef castFailed() { return ""; }
};

template <>
struct CastInfo<StringRef, const M::Setting *>
    : public ConstStrippingForwardingCast<StringRef, const M::Setting *,
                                          CastInfo<StringRef, M::Setting *>> {};

/// It would be possible to have strongly-typed configs as well - the simplest
/// model would be to have a CastInfo<T, M::Setting> that simply parses T from a
/// string at the time of the cast - a more complex method would be to actually
/// modify M::Config to have a structure more like M::EntitlementStore so that
/// configs can be parsed into structures upon reading the file.
} // namespace llvm

#endif // SUPPORT_SETTINGS_SETTING_H
