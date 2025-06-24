//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "WebUtilities.h"
#include "motr/Log.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/fetch.h>
#endif

namespace M::motr::Gui {

struct FetchCallbackData {
  std::function<void(std::string_view)> onSuccess;
  std::function<void(int, const char *)> onError;
};

#ifdef __EMSCRIPTEN__
static void fetchSuccessCallback(emscripten_fetch_t *fetch) {
  auto *data = static_cast<FetchCallbackData *>(fetch->userData);
  std::string_view response{fetch->data, size_t(fetch->numBytes)};
  data->onSuccess(response);
  emscripten_fetch_close(fetch);
  delete data;
}

static void fetchErrorCallback(emscripten_fetch_t *fetch) {
  auto *data = static_cast<FetchCallbackData *>(fetch->userData);
  data->onError(fetch->status, fetch->statusText);
  emscripten_fetch_close(fetch);
  delete data;
}
#endif

void fetchUrlAsync(const std::string &url,
                   std::function<void(std::string_view)> onSuccess,
                   std::function<void(int, const char *)> onError) {
#ifdef __EMSCRIPTEN__
  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);
  strcpy(attr.requestMethod, "GET");
  attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;

  auto *data = new FetchCallbackData{std::move(onSuccess), std::move(onError)};
  attr.userData = data;
  attr.onsuccess = fetchSuccessCallback;
  attr.onerror = fetchErrorCallback;

  emscripten_fetch(&attr, url.c_str());
#endif
}

void asyncLoadJsonUrl(std::string_view url,
                      std::function<bool(nlohmann::json)> onSuccess) {
  fetchUrlAsync(
      std::string(url),
      [onSuccess](std::string_view response) {
        auto json = nlohmann::json::parse(response);
        onSuccess(json);
      },
      [](int status, const char *statusText) {
        MOTR_LOG("Error fetching layout: status={} statusText={}", status,
                 statusText);
      });
}

std::string eval(std::string_view cmd) {
#ifdef __EMSCRIPTEN__
  // clang-format off
  char *result = reinterpret_cast<char *>(EM_ASM_INT(
      {
        var result = eval(UTF8ToString($0));
        var lengthBytes = lengthBytesUTF8(result) + 1;
        var stringOnWasmHeap = _malloc(lengthBytes);
        stringToUTF8(result, stringOnWasmHeap, lengthBytes);
        return stringOnWasmHeap;
      },
      cmd.data()));
  // clang-format on

  std::string strResult(result);
  free(result);
  return strResult;
#else
  return "";
#endif
}

std::unordered_map<std::string, std::string> &getURLSearchParams() {
  static std::unordered_map<std::string, std::string> cachedParams;
  static bool isCached = false;

  if (isCached)
    return cachedParams;

  std::string search = eval("window.location.search");

  if (!search.empty() && search[0] == '?') {
    search = search.substr(1);
  }

  size_t start = 0;
  while (start < search.length()) {
    size_t end = search.find('&', start);
    if (end == std::string::npos)
      end = search.length();

    size_t sep = search.find('=', start);
    if (sep != std::string::npos && sep < end) {
      std::string key = search.substr(start, sep - start);
      std::string value = search.substr(sep + 1, end - sep - 1);
      cachedParams[key] = value;
    }

    start = end + 1;
  }

  isCached = true;
  return cachedParams;
}

int getWebsocketPort(int idx) {
  auto &params = getURLSearchParams();
  std::string key = idx ? fmt::format("port{}", idx) : "port";
  auto it = params.find(key);

  if (it == params.end()) {
    return 0;
  }

  if (it != params.end()) {
    const char *str = it->second.c_str();
    char *end;
    long port = strtol(str, &end, 10);
    if (end == str || *end != '\0' || port < 0 || port > 65535) {
      return 8888;
    }
    return static_cast<int>(port);
  }
  return 8888;
}

} // namespace M::motr::Gui
