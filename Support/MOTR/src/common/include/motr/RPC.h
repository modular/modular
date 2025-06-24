//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_RPC_H
#define MOTR_RPC_H

#include <iostream>
#include <string>
#include <tuple>
#include <utility>

#include "motr/Constants.h"
#include "motr/Hash.h"

#define STRINGIFY(x) #x

// Macro to reflect a single field
#define RPC_REFLECT_FIELD(f, field) f(STRINGIFY(field), field)

#define ARG_COUNT_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define ARG_COUNT(...)                                                         \
  ARG_COUNT_IMPL(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define RPC_DECLARE_FINGERPRINT(...)                                           \
  static constexpr uint64_t getRPCFingerprint() {                              \
    return ::M::motr::RPC::computeRPCFingerprint(__VA_ARGS__);                 \
  }
// Concatenation macros
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define CONCAT_IMPL(a, b) a##b

// Individual RPC_REFLECTABLE macros for each count
#define RPC_REFLECTABLE_1(f1)                                                  \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(STRINGIFY(f1));                    \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
  }

#define RPC_REFLECTABLE_2(f1, f2)                                              \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(STRINGIFY(f1), STRINGIFY(f2));     \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
  }

#define RPC_REFLECTABLE_3(f1, f2, f3)                                          \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(STRINGIFY(f1), STRINGIFY(f2),      \
                                            STRINGIFY(f3));                    \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
  }

#define RPC_REFLECTABLE_4(f1, f2, f3, f4)                                      \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(STRINGIFY(f1), STRINGIFY(f2),      \
                                            STRINGIFY(f3), STRINGIFY(f4));     \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
  }

#define RPC_REFLECTABLE_5(f1, f2, f3, f4, f5)                                  \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(STRINGIFY(f1), STRINGIFY(f2),      \
                                            STRINGIFY(f3), STRINGIFY(f4),      \
                                            STRINGIFY(f5));                    \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
  }

#define RPC_REFLECTABLE_6(f1, f2, f3, f4, f5, f6)                              \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(STRINGIFY(f1), STRINGIFY(f2),      \
                                            STRINGIFY(f3), STRINGIFY(f4),      \
                                            STRINGIFY(f5), STRINGIFY(f6));     \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
  }

#define RPC_REFLECTABLE_7(f1, f2, f3, f4, f5, f6, f7)                          \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(                                   \
          STRINGIFY(f1), STRINGIFY(f2), STRINGIFY(f3), STRINGIFY(f4),          \
          STRINGIFY(f5), STRINGIFY(f6), STRINGIFY(f7));                        \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
  }

#define RPC_REFLECTABLE_8(f1, f2, f3, f4, f5, f6, f7, f8)                      \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(                                   \
          STRINGIFY(f1), STRINGIFY(f2), STRINGIFY(f3), STRINGIFY(f4),          \
          STRINGIFY(f5), STRINGIFY(f6), STRINGIFY(f7), STRINGIFY(f8));         \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
    RPC_REFLECT_FIELD(f, f8);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
    RPC_REFLECT_FIELD(f, f8);                                                  \
  }

#define RPC_REFLECTABLE_9(f1, f2, f3, f4, f5, f6, f7, f8, f9)                  \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(                                   \
          STRINGIFY(f1), STRINGIFY(f2), STRINGIFY(f3), STRINGIFY(f4),          \
          STRINGIFY(f5), STRINGIFY(f6), STRINGIFY(f7), STRINGIFY(f8),          \
          STRINGIFY(f9));                                                      \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
    RPC_REFLECT_FIELD(f, f8);                                                  \
    RPC_REFLECT_FIELD(f, f9);                                                  \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
    RPC_REFLECT_FIELD(f, f8);                                                  \
    RPC_REFLECT_FIELD(f, f9);                                                  \
  }

#define RPC_REFLECTABLE_10(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10)            \
  using IsReflectable = void;                                                  \
  static constexpr uint64_t RPCFingerprint =                                   \
      ::M::motr::RPC::computeRPCFingerprint(                                   \
          STRINGIFY(f1), STRINGIFY(f2), STRINGIFY(f3), STRINGIFY(f4),          \
          STRINGIFY(f5), STRINGIFY(f6), STRINGIFY(f7), STRINGIFY(f8),          \
          STRINGIFY(f9), STRINGIFY(f10));                                      \
  template <typename F>                                                        \
  void reflect(F &&f) {                                                        \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
    RPC_REFLECT_FIELD(f, f8);                                                  \
    RPC_REFLECT_FIELD(f, f9);                                                  \
    RPC_REFLECT_FIELD(f, f10);                                                 \
  }                                                                            \
  template <typename F>                                                        \
  void reflect(F &&f) const {                                                  \
    RPC_REFLECT_FIELD(f, f1);                                                  \
    RPC_REFLECT_FIELD(f, f2);                                                  \
    RPC_REFLECT_FIELD(f, f3);                                                  \
    RPC_REFLECT_FIELD(f, f4);                                                  \
    RPC_REFLECT_FIELD(f, f5);                                                  \
    RPC_REFLECT_FIELD(f, f6);                                                  \
    RPC_REFLECT_FIELD(f, f7);                                                  \
    RPC_REFLECT_FIELD(f, f8);                                                  \
    RPC_REFLECT_FIELD(f, f9);                                                  \
    RPC_REFLECT_FIELD(f, f10);                                                 \
  }

// Main RPC_REFLECTABLE macro
#define RPC_REFLECTABLE(...)                                                   \
  CONCAT(RPC_REFLECTABLE_, ARG_COUNT(__VA_ARGS__))(__VA_ARGS__)

namespace M::motr::RPC {

template <typename T, typename = void>
struct is_rpc_reflectable : std::false_type {};

template <typename T>
struct is_rpc_reflectable<T, std::void_t<typename T::IsReflectable>>
    : std::true_type {};

template <typename... Args>
constexpr uint64_t computeRPCFingerprint(Args... strings) {
  Hash::Value hashes[] = {Hash::Value{strings}...};
  uint64_t x0r = 0;
  int64_t sum = 0;

  for (auto &hash : hashes) {
    x0r ^= hash.v;
    sum += hash.v;
  }

  constexpr const Hash::Value injectList[] = {
      Constants::__rpc_fingerprint__::hash,
      Constants::__rpc_request_id__::hash,
  };

  // Inject keys into the hash as all RPCResult sets will have this key
  for (auto &key : injectList) {
    x0r ^= key.v;
    sum += key.v;
  }

  return (x0r + sum) ^ sum;
}

} // namespace M::motr::RPC

#endif // MOTR_RPC_H
