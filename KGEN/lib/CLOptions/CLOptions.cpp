//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

//===--------------------------------------------------------------------===//
// getHostCPUFeatures
//===--------------------------------------------------------------------===//

std::string KGEN::getHostCPUFeatures() {
  llvm::StringMap<bool> hostFeatures;

  // Get the host features.
  std::string featureStr;
  llvm::raw_string_ostream os(featureStr);
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    llvm::interleave(
        llvm::make_filter_range(hostFeatures, [](auto &f) { return f.second; }),
        os, [&](auto &f) { os << '+' << f.first(); }, ",");
  }

  return featureStr;
}

//===--------------------------------------------------------------------===//
// CommandLineFunc implementation
//===--------------------------------------------------------------------===//

ErrorOrSuccess
CommandLineFunc::verifyFuncSignature(mlir::FunctionType funcType) const {
  if (signature == "f32()") {
    if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1 ||
        !funcType.getResult(0).isa<Float32Type>()) {
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   mlir::debugString(funcType) + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "()") {
    if (funcType.getNumResults() != 0 || funcType.getNumInputs() != 0) {
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   mlir::debugString(funcType) + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "index()") {
    if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1 ||
        !funcType.getResult(0).isa<IndexType>()) {
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   mlir::debugString(funcType) + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "f32(f32)") {
    if (funcType.getNumInputs() != 1 || funcType.getNumResults() != 1 ||
        !funcType.getResult(0).isa<Float32Type>() ||
        !funcType.getInput(0).isa<Float32Type>()) {
      std::string ktype;
      llvm::raw_string_ostream os(ktype);
      os << funcType;
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   ktype + ", but got " + signature);
    }
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

ErrorOrSuccess
CommandLineFunc::executeAndPrint(KGEN::CompiledFunc &compiledFunc) const {
  if (signature == "f32()") {
    printf("--- '%s' returned %f\n", name.c_str(),
           compiledFunc.invoke<float>());
    return M::success();
  } else if (signature == "index()") {
    printf("--- '%s' returned %ld\n", name.c_str(),
           compiledFunc.invoke<ssize_t>());
    return M::success();
  } else if (signature == "()") {
    compiledFunc.invoke<void>();
    printf("--- '%s' finished\n", name.c_str());
    return M::success();
  } else if (signature == "f32(f32)") {
    // TODO: We could parse a float for this, but for now just pass in 1.0 for
    //       all floats.
    printf("--- '%s' returned %f\n", name.c_str(),
           compiledFunc.invoke<float, float>(1.0));
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

bool CommandLineFuncParser::parse(llvm::cl::Option &o, StringRef argName,
                                  StringRef argValue, CommandLineFunc &val) {
  // Match a function name and signature, of the form: `name:signature`. This
  // check also ensures that "name" supports '::' tokens, which may be used for
  // scope signifiers.
  static llvm::Regex funcAndSignatureMatcher("(.*[^:]):([^:].*)");

  // Check if the value contains the name and signature.
  SmallVector<StringRef> matches;
  if (funcAndSignatureMatcher.match(argValue, &matches)) {
    val.name = matches[1];
    val.signature = matches.back();
    return false;
  }

  // Otherwise, if we don't have a signature, the value is the name.
  val.name = argValue;
  return false;
}

//===--------------------------------------------------------------------===//
// TraceProfiler
//===--------------------------------------------------------------------===//

TraceProfiler::TraceProfiler(const KGENCommonOptions &clOptions) {
  if (!clOptions.timeTrace)
    return;
  profiler.emplace(clOptions.timeTraceGranularity, "kgen");

  std::error_code ec;
  std::filesystem::path derived = std::filesystem::absolute(
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);
  if (ec)
    clOptions.reportError("cannot get the modular derived path: " +
                          ec.message());

  outputFilePath = derived / "kgen.trace.json";
}

TraceProfiler::~TraceProfiler() {
  if (!profiler)
    return;
  if (auto err = profiler->write(outputFilePath.string(), "-"))
    llvm::errs() << "unable to write trace file: " << err.getError();
}
