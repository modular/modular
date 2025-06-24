//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/Log.h"
#include "motr/Mailbox.h"
#include "motr/Tags.h"
#include "motr/motr.h"
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace M::motr;
using Outbox = ServerOutboxString;
using StringViews = Outbox::StringViews;

static StringViews argsToStringViews(int argc, char **argv) {
  StringViews views;
  views.reserve(argc);
  for (int i = 0; i < argc; i++)
    views.emplace_back(argv[i]);
  return views;
}

static void logStringViews(const StringViews &views) {
  for (auto &sv : views) {
    Hash::Value hash{sv};
    MOTR_LOG("[0x{:016x}] [{:3d}] \"{}\"", hash.v, sv.size(), sv);
  }
}

std::optional<uint64_t> parseAsUint64(std::string_view str) {
  uint64_t result;
  auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
  if (ec == std::errc())
    return result;
  return std::nullopt;
}

std::pair<std::string_view, std::string_view>
splitKeyVal(std::string_view str) {
  auto pos = str.find('=');
  if (pos == std::string_view::npos) {
    return std::make_pair("", "");
  }
  return std::make_pair(str.substr(0, pos), str.substr(pos + 1));
}

std::string truncValue(std::string_view str) {
  if (str.size() > 10) {
    return std::string(str.substr(0, 10)) + "...";
  }
  return std::string(str);
}

int tagsMain(int argc, char **argv) {
  MOTR_TraceProgram(tagsMainTrace, "motr tags");

  // todo: add an option to wait for the string msg to be consumed
  if (argc < 3) {
    MOTR_LOG("Usage: motr tags <key1>=<value1> ... <keyN>=<valueN>", "");
    return 1;
  }

  // shift "motr tags" out of the way
  argc -= 2;
  argv += 2;

  Span<MessageType::RPCCall> rpcCall;

  StringViews stringViews = argsToStringViews(argc, argv);
  StringViews sendStringViews;
  sendStringViews.reserve(stringViews.size() * 2);
  for (auto &sv : stringViews) {
    std::pair<std::string_view, std::string_view> kv = splitKeyVal(sv);
    if (kv.first.empty()) {
      MOTR_LOG("Invalid tag: {}", sv);
      return 1;
    }
    sendStringViews.emplace_back(kv.first);
    sendStringViews.emplace_back(kv.second);
  }

  Outbox::send(sendStringViews);

  for (int i = 0; i < sendStringViews.size(); i += 2) {
    auto key = sendStringViews[i];
    auto val = sendStringViews[i + 1];
    MOTR_LOG("{}={}", key, truncValue(val));

    if (auto optionalValAsUint64 = parseAsUint64(val); optionalValAsUint64) {
      TagInt{Hash::Value{key}.v, optionalValAsUint64.value()};
    } else {
      TagStr{Hash::Value{key}.v, Hash::Value{val}.v};
    }
  }

  return 0;
}
