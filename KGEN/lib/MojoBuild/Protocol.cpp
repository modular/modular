//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoBuild/Protocol.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

using namespace M;
using namespace M::Build;

// Helper that doesn't treat `null` and absent fields as failures.
// FIXME(MOTO-420): This is copied from MLIR, and should be shared instead.
template <typename T>
static bool mapOptOrNull(const llvm::json::Value &params,
                         llvm::StringLiteral prop, T &out,
                         llvm::json::Path path) {
  const llvm::json::Object *o = params.getAsObject();
  assert(o);

  // Field is missing or null.
  auto *v = o->get(prop);
  if (!v || v->getAsNull())
    return true;
  return fromJSON(*v, out, path.field(prop));
}

//===----------------------------------------------------------------------===//
// build/initialize request params
//===----------------------------------------------------------------------===//

llvm::json::Value M::Build::toJSON(const InitializeBuildParams &value) {
  return llvm::json::Object{{"displayName", value.displayName},
                            {"version", value.version},
                            {"bspVersion", value.bspVersion},
                            {"rootUri", value.rootUri}};
}

bool M::Build::fromJSON(const llvm::json::Value &value,
                        InitializeBuildParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("displayName", result.displayName) &&
         o.map("version", result.version) &&
         o.map("bspVersion", result.bspVersion) &&
         o.map("rootUri", result.rootUri);
}

//===----------------------------------------------------------------------===//
// build/initialize request result
//===----------------------------------------------------------------------===//

llvm::json::Value M::Build::toJSON(const CompileProvider &value) {
  return llvm::json::Object{{"languageIds", value.languageIds}};
}

bool M::Build::fromJSON(const llvm::json::Value &value, CompileProvider &result,
                        llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("languageIds", result.languageIds);
}

llvm::json::Value M::Build::toJSON(const BuildServerCapabilities &value) {
  llvm::json::Object result;
  if (value.compileProvider)
    result["compileProvider"] = value.compileProvider;
  return std::move(result);
}

bool M::Build::fromJSON(const llvm::json::Value &value,
                        BuildServerCapabilities &result,
                        llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o &&
         mapOptOrNull(value, "compileProvider", result.compileProvider, path);
}

llvm::json::Value M::Build::toJSON(const InitializeBuildResult &value) {
  return llvm::json::Object{{"displayName", value.displayName},
                            {"version", value.version},
                            {"bspVersion", value.bspVersion},
                            {"capabilities", value.capabilities}};
}

bool M::Build::fromJSON(const llvm::json::Value &value,
                        InitializeBuildResult &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("displayName", result.displayName) &&
         o.map("version", result.version) &&
         o.map("bspVersion", result.bspVersion) &&
         o.map("capabilities", result.capabilities);
}

//===----------------------------------------------------------------------===//
// buildTarget/compile request params
//===----------------------------------------------------------------------===//

llvm::json::Value M::Build::toJSON(const BuildTargetIdentifier &value) {
  return llvm::json::Object{{"uri", value.uri}};
}

bool M::Build::fromJSON(const llvm::json::Value &value,
                        BuildTargetIdentifier &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri);
}

llvm::json::Value M::Build::toJSON(const CompileParams &value) {
  llvm::json::Object result{{"targets", value.targets}};
  if (value.originId)
    result["originId"] = *value.originId;
  if (value.arguments)
    result["arguments"] = llvm::json::Array(*value.arguments);
  return std::move(result);
}

bool M::Build::fromJSON(const llvm::json::Value &value, CompileParams &result,
                        llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("targets", result.targets) &&
         mapOptOrNull(value, "originId", result.originId, path) &&
         mapOptOrNull(value, "arguments", result.arguments, path);
}

//===----------------------------------------------------------------------===//
// buildTarget/compile request result
//===----------------------------------------------------------------------===//

llvm::json::Value M::Build::toJSON(const CompileResult &value) {
  llvm::json::Object result;
  if (value.originId)
    result["originId"] = *value.originId;
  result["statusCode"] = static_cast<int>(value.statusCode);
  return std::move(result);
}

bool M::Build::fromJSON(const llvm::json::Value &value, CompileResult &result,
                        llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  int statusCode;
  if (!o || !mapOptOrNull(value, "originId", result.originId, path) ||
      !o.map("statusCode", statusCode))
    return false;
  result.statusCode = static_cast<StatusCode>(statusCode);
  return true;
}

//===----------------------------------------------------------------------===//
// build/shutdown request
//===----------------------------------------------------------------------===//

llvm::json::Value M::Build::toJSON(const NoParams &) { return nullptr; }

bool M::Build::fromJSON(const llvm::json::Value &, NoParams &,
                        llvm::json::Path) {
  return true;
}
