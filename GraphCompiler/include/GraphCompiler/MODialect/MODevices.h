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

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H

#include "GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/PagedVector.h"
namespace M::MO {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Returns the #M.device_ref annotation for op, or the 'host' device
/// if there is no such annotation on the op itself.
DeviceRefAttr getDeviceOrDefault(MOHasDeviceInterface op);

/// As above, but for any operation.
DeviceRefAttr getDeviceOrDefaultAny(Operation *op);

/// Returns the #M.device_ref annotation for type, or the 'host' device if
/// there is none.
DeviceRefAttr getDeviceOrDefault(MOHasDeviceTypeInterface type);

/// As above, but for any type.
DeviceRefAttr getDeviceOrDefaultAny(Type type);

/// Returns all devices used by the inputs or outputs of a given MO op, with
/// duplicates removed.
SmallVector<DeviceRefAttr> getUniqueDevicesFromKernel(Operation *op);

// (See also getHostDevice, declared in MOOps.h so can be used in MOOps.td)

/// Returns true if op should execute on the 'host'.
bool isOnHost(Operation *op);

/// Return true if a value of type should be stored on the 'host'.
bool isOnHost(Type type);

//===----------------------------------------------------------------------===//
// DeviceAnnots
//===----------------------------------------------------------------------===//

/// Captures all device-related annotations from one or more operations, and
/// allows them to be 'stamped' onto new ops.
struct DeviceAnnots {
  /// The 'device' attribute of the original op, or null.
  DeviceRefAttr device;

  /// Returns the device annotations for operation.
  static DeviceAnnots forOp(Operation *op);

  /// Unions the device annotations of this and that into this. Returns false if
  /// this and that have conflicting annotations.
  bool add(const DeviceAnnots &that);

  /// Captures the device annotations for operation in this. Returns false if
  /// this and op have conflicting annotations. No-op if op is null.
  bool addOp(Operation *op);

  /// Stamps these device annotations into operation and returns it.
  Operation *stampAny(Operation *op) const;

  /// Stamps these device annotations into op, if it takes an explicit device
  /// annotation, and returns it.
  template <typename Op>
  Op stamp(Op op) const {
    if (op.template hasTrait<MOExplicitDevice>() || isa<CustomOp>(op))
      stampAny(op);
    return op;
  }

  /// Return true if these annotations are consistent with the given device,
  /// which may be null.
  bool isConsistentWith(DeviceRefAttr thatDevice) const;

  /// Returns true if this and that agree in their annotations where they
  /// are given.
  bool isConsistentWith(const DeviceAnnots &that) const;
};

//===----------------------------------------------------------------------===//
// DeviceAwareRewriter
//===----------------------------------------------------------------------===//

/// Wraps a RewriterBase to automatically collect and stamp ops with device
/// annotations.
///
/// Wraps rather than derives since RewriterBase is extensible; only a dozen or
/// so methods need forwarding.
struct DeviceAwareRewriter {
  /// The accumulated device annotations for all the matched ops.
  DeviceAnnots deviceAnnots;
  mlir::RewriterBase &rewriter;
  /// Ops which may be bound by patterns, besides the constructor's anchor op.
  /// A PagedVector keeps the addresses returned by `pushOp()` stable as it
  /// grows.
  llvm::PagedVector<Operation *, sizeof(Operation *) * 16> matchedOps;

  /// Captures the annotations of `op`, which the rewriter is assumed to
  /// replace.
  DeviceAwareRewriter(Operation *op, mlir::RewriterBase &rewriter)
      : deviceAnnots(DeviceAnnots::forOp(op)), rewriter(rewriter) {}

  /// Pushes an operation onto matchedOps for later checking by addOps.
  void pushOp(Operation *op) {
    matchedOps.resize(matchedOps.size() + 1);
    matchedOps[matchedOps.size() - 1] = op;
  }

  /// Pushes a null operation onto matchedOps and returns its address, for an
  /// op-matching pattern to assign on match.
  Operation **pushOp() {
    matchedOps.resize(matchedOps.size() + 1);
    return &matchedOps[matchedOps.size() - 1];
  }

  /// Captures the device annotations of matchedOps and `ops`, which the
  /// rewriter is assumed to replace. Returns false on conflicting annotations.
  /// Null ops are ignored.
  [[nodiscard]] bool
  addOps(const std::initializer_list<Operation *> &ops = {}) {
    return llvm::all_of(
               matchedOps.materialized(),
               [this](Operation *op) { return deviceAnnots.addOp(op); }) &&
           llvm::all_of(
               ops, [this](Operation *op) { return deviceAnnots.addOp(op); });
  }

  /// Like OpBuilder::create, but the new op is stamped.
  ///
  /// TODO: Every created/replaced op is stamped, which is only correct while
  /// MO ops take and return values on their own device. Relaxing that needs
  /// rewrites to tell ops that must be stamped from ones that need not (eg a
  /// mo.constant for an index operand).
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    return deviceAnnots.stamp(
        OpTy::create(rewriter, location, std::forward<Args>(args)...));
  }

  /// Create a constant stamped with the host device, for operands constrained
  /// to the host.
  template <typename... Args>
  ConstantOp createHostConstant(Location location, Args &&...args) {
    return ConstantOp::create(rewriter, location, std::forward<Args>(args)...,
                              getHostDevice(rewriter.getContext()));
  }

  /// Create a constant whose op and tensor type are stamped with the current
  /// device.
  template <typename... Args>
  ConstantOp createDeviceConstant(Location location, Args &&...args) {
    return ConstantOp::create(rewriter, location, std::forward<Args>(args)...,
                              deviceAnnots.device);
  }

  template <typename CallableT>
  void modifyOpInPlace(Operation *root, CallableT &&callable) {
    rewriter.modifyOpInPlace(root, std::forward<CallableT>(callable));
  }

  /// Like RewriterBase::replaceOpWithNewOp, but the new op is stamped.
  template <typename OpTy, typename... Args>
  OpTy replaceOpWithNewOp(Operation *op, Args &&...args) {
    auto newOp =
        create<OpTy, Args...>(op->getLoc(), std::forward<Args>(args)...);
    rewriter.replaceOp(op, newOp.getOperation());
    return newOp;
  }

  //===--------------------------------------------------------------------===//
  /// The following redirect to the underlying rewriter.
  //===--------------------------------------------------------------------===//

  template <typename ArgT>
  LogicalResult notifyMatchFailure(ArgT &&arg, const Twine &msg) {
    return rewriter.notifyMatchFailure(std::forward<ArgT>(arg), msg);
  }
  IntegerAttr getIndexAttr(int64_t value) {
    return rewriter.getIndexAttr(value);
  }
  MLIRContext *getContext() { return rewriter.getContext(); }
  TypedAttr getDimAttr(int64_t value) {
    return MOSH::ShapeAttr::getDim(rewriter.getContext(), value);
  }
  IntegerType getIntegerType(unsigned width, bool isSigned) {
    return rewriter.getIntegerType(width, isSigned);
  }
  template <typename Ty, typename... Args>
  Ty getType(Args &&...args) {
    return rewriter.getType<Ty>(std::forward<Args>(args)...);
  }
  template <typename Attr, typename... Args>
  Attr getAttr(Args &&...args) {
    return rewriter.getAttr<Attr>(std::forward<Args>(args)...);
  }
  void setInsertionPoint(Operation *op) { rewriter.setInsertionPoint(op); }
  void replaceOp(Operation *op, ValueRange newValues) {
    rewriter.replaceOp(op, newValues);
  }
  void replaceOp(Operation *op, Operation *newOp) {
    rewriter.replaceOp(op, newOp);
  }
  void replaceAllUsesWith(Value from, Value to) {
    rewriter.replaceAllUsesWith(from, to);
  }
  void replaceAllUsesWith(ValueRange from, ValueRange to) {
    rewriter.replaceAllUsesWith(from, to);
  }
  void replaceAllOpUsesWith(Operation *op, ValueRange to) {
    rewriter.replaceAllOpUsesWith(op, to);
  }
  void replaceAllOpUsesWith(Operation *op, Operation *to) {
    rewriter.replaceAllOpUsesWith(op, to);
  }
  void setInsertionPointAfter(Operation *op) {
    rewriter.setInsertionPointAfter(op);
  }
  void replaceAllUsesExcept(Value from, Value to, Operation *exceptedUser) {
    rewriter.replaceAllUsesExcept(from, to, exceptedUser);
  }
  Operation *clone(Operation &op) { return rewriter.clone(op); }
  void eraseOp(Operation *op) { rewriter.eraseOp(op); }
  Operation *create(const OperationState &state) {
    return rewriter.create(state);
  }
  IndexType getIndexType() { return rewriter.getIndexType(); }
  BoolAttr getBoolAttr(bool value) { return rewriter.getBoolAttr(value); }
  NamedAttribute getNamedAttr(StringRef name, Attribute val) {
    return rewriter.getNamedAttr(name, val);
  }
  DictionaryAttr getDictionaryAttr(ArrayRef<NamedAttribute> value) {
    return rewriter.getDictionaryAttr(value);
  }
  DenseIntElementsAttr getI64VectorAttr(ArrayRef<int64_t> values) {
    return rewriter.getI64VectorAttr(values);
  }
};

namespace detail {

//===----------------------------------------------------------------------===//
// DeviceAwarePattern
//===----------------------------------------------------------------------===//

/// TODO: clean this up, ask C++ geniuses
/// A drop-in replacement for `OpRewritePattern`, `OpInterfaceRewritePattern`
/// and the like that propagates device annotations via `DeviceAwareRewriter`.
/// For example, `RewritePatternT` = `OpRewritePattern<AddOp>`, `OpT` = `AddOp`.
///
/// ```
/// struct MyRewrite :
///   public DeviceAwarePattern<OpInterfaceRewritePattern, Reduction>  {
///
///   LogicalResult
///   deviceAwareMatchAndRewrite(Reduction op,
///                              DeviceAwareRewriter &commonRewriter)
///                              const override {
///     ...
///   }
/// ...
/// };
/// ```
template <typename RewritePatternT, typename OpT>
struct DeviceAwarePatternShim : public RewritePatternT {
  using RewritePatternT::RewritePatternT;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &commonRewriter) const final {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    return deviceAwareMatchAndRewrite(op, rewriter);
  }

  virtual LogicalResult
  deviceAwareMatchAndRewrite(OpT op, DeviceAwareRewriter &rewriter) const = 0;
};

} // namespace detail

/// Forwards template parameters to `RewritePatternBaseT` for nicer syntax.
template <template <typename> typename RewritePatternBaseT, typename OpT>
struct DeviceAwareRewritePattern
    : detail::DeviceAwarePatternShim<RewritePatternBaseT<OpT>, OpT> {
  using detail::DeviceAwarePatternShim<RewritePatternBaseT<OpT>,
                                       OpT>::DeviceAwarePatternShim;
};

//===----------------------------------------------------------------------===//
// DeviceInfoMapping
//===----------------------------------------------------------------------===//

/// Reads the `mo.device_info_mapping` DictionaryAttr from a ModuleOp and
/// provides lookup from DeviceRefAttr to DeviceInfoAttr.
class DeviceInfoMapping {
public:
  static constexpr llvm::StringLiteral kAttrName = "mo.device_info_mapping";

  explicit DeviceInfoMapping(mlir::ModuleOp moduleOp);

  /// Returns null if the device has no entry.
  DeviceInfoAttr get(DeviceRefAttr device) const;

private:
  llvm::StringMap<DeviceInfoAttr> _mapping;
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H
