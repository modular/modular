//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/PlatformPredicate.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::MOGG;
using namespace mlir;

namespace M::MOGG {

#define GEN_PASS_DEF_TESTKERNELREGISTRATIONS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

constexpr llvm::StringLiteral kTestDeviceInfosAttrName =
    "mogg.test_device_infos";

llvm::StringRef getOrderName(PlatformPredicateOrder order) {
  switch (order) {
  case PlatformPredicateOrder::Equivalent:
    return "equivalent";
  case PlatformPredicateOrder::FirstNarrower:
    return "first narrower";
  case PlatformPredicateOrder::SecondNarrower:
    return "second narrower";
  case PlatformPredicateOrder::Incomparable:
    return "incomparable";
  }
  llvm_unreachable("unknown platform order");
}

struct TestKernelRegistrationsPass
    : public impl::TestKernelRegistrationsBase<TestKernelRegistrationsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto devices = module->getAttrOfType<ArrayAttr>(kTestDeviceInfosAttrName);
    llvm::SmallVector<DeviceInfoAttr> deviceInfos;
    if (devices) {
      for (Attribute element : devices) {
        auto deviceInfo = dyn_cast<DeviceInfoAttr>(element);
        if (!deviceInfo) {
          module.emitError()
              << kTestDeviceInfosAttrName
              << " must hold only device_info attributes, found " << element;
          return signalPassFailure();
        }
        deviceInfos.push_back(deviceInfo);
      }
    }

    // Registrations that have at least one `cond=` decl, with all their decls.
    llvm::MapVector<StringAttr, SmallVector<KernelDeclOp>> registrations;
    for (KernelDeclOp decl : module.getOps<KernelDeclOp>())
      registrations[decl.getRegistrationAttr()].push_back(decl);
    registrations.remove_if([](const auto &entry) {
      return llvm::none_of(entry.second, [](KernelDeclOp decl) {
        return static_cast<bool>(decl.getPlatformPredicateAttr());
      });
    });

    for (auto &[registration, decls] : registrations) {
      for (KernelDeclOp decl : decls) {
        ErrorOr<PlatformPredicate> platform =
            getRegistrationPlatformPredicate(decl);
        llvm::errs() << "platform @" << decl.getSymName() << " = ";
        if (platform.isError()) {
          llvm::errs() << "error: " << platform.getError() << "\n";
          continue;
        }
        platform->print(llvm::errs());
        llvm::errs() << "\n";
        for (DeviceInfoAttr device : deviceInfos) {
          llvm::errs() << "platform @" << decl.getSymName() << " on "
                       << Attribute(device) << " = "
                       << (platform->holdsFor(getDeviceFields(device))
                               ? "true"
                               : "false")
                       << "\n";
        }
      }

      for (size_t i = 0; i < decls.size(); ++i) {
        for (size_t j = i + 1; j < decls.size(); ++j) {
          ErrorOr<PlatformPredicate> first =
              getRegistrationPlatformPredicate(decls[i]);
          ErrorOr<PlatformPredicate> second =
              getRegistrationPlatformPredicate(decls[j]);
          if (first.isError() || second.isError())
            continue;
          PlatformPredicateComparison comparison =
              comparePlatformPredicates(*first, *second);
          llvm::errs() << "compare @" << decls[i].getSymName() << " @"
                       << decls[j].getSymName() << " = "
                       << getOrderName(comparison.order) << "\n";
        }
      }
    }

    auto mapping =
        module->getAttrOfType<DictionaryAttr>(MO::DeviceInfoMapping::kAttrName);
    if (!mapping)
      return;
    MojoRegistry registry(module);
    for (auto &entry : registrations) {
      StringAttr registration = entry.first;
      for (NamedAttribute device : mapping) {
        StringRef label = device.getName().getValue();
        auto deviceRef = DeviceRefAttr::get(module.getContext(), label, 0);
        ErrorOr<KernelDeclOp> decl =
            registry.getResolvedKernel(registration.getValue(), deviceRef);
        llvm::errs() << "resolve \"" << registration.getValue() << "\" on \""
                     << label << "\" -> ";
        if (decl.isError())
          llvm::errs() << "error: " << decl.getError();
        else
          llvm::errs() << "@" << decl.takeValue().getSymName();
        llvm::errs() << "\n";
      }
    }
  }
};

} // namespace

} // namespace M::MOGG
