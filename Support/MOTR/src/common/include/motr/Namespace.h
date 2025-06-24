//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_NAMESPACE_H
#define MOTR_NAMESPACE_H

#include "motr/Common.h"

#include <cassert>
#include <cstdlib>
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <string>
#include <vector>

namespace M::motr {
struct Namespace;
}

struct M::motr::Namespace {
  static const constexpr char *DefaultNamespace = "motr";
  static const constexpr char *EnvVar = "MOTR_NAMESPACE";
  static const constexpr char *sep = "_";

  template <typename... Args>
  static std::string join(Args &&...args) {
    std::vector<std::string> argList = {
        fmt::format("{}", std::forward<Args>(args))...};
    return fmt::format("{}", fmt::join(argList, sep));
  }

  template <typename... Args>
  static std::string makeSHMName(Args &&...args) {
    std::string str = "/" + join("modular", get(), std::forward<Args>(args)...);

#ifdef MOTR_PLATFORM_MACOS
    // IMPORTANT: On MacOS, the eventual string used for shared memory
    // MUST be < 31 chars
    assert(str.size() <= 31 && "Shared memory name is too long");
#endif

    return str;
  }

  static std::string getDefault() {
    const char *env = getenv(EnvVar);
    return env ? env : DefaultNamespace;
  }

  static std::string &reset() {
    set(getDefault());
    return get();
  }

  static std::string &get() {
    static std::string str = getDefault();
    return str;
  }
  static int &generation() {
    // generation starts at 1 to because
    // the static init in get() will
    // init to the env or default value on demand
    static int generation = 1;
    return generation;
  }

  static int set(std::string_view newValue) {
    get() = newValue;
    return ++generation();
  }
}; // struct M::motr::Namespace

#endif // MOTR_NAMESPACE_H
