//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
#define SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H

#include "Support/Compiler/ErrorTree.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpDefinition.h"

namespace M {
class InterpreterState;
class TargetInfoAttr;

using InterpretHook = ErrorTreeOrSuccess (*)(Operation *, ArrayRef<Attribute>,
                                             const void *, InterpreterState &);
using GenBytecodeHook = ErrorOrSuccess (*)(Operation *, void *, TargetInfoAttr);

struct OpBytecodeGenerator {
  uint32_t payloadSize;
  GenBytecodeHook genBytecode;
  InterpretHook interpret;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterOpInterface.h.inc"
#include "KGEN/Interpreter/MemoryableTypeInterface.h.inc"

//===----------------------------------------------------------------------===//
// Delegate Interace Declarations
//===----------------------------------------------------------------------===//

namespace M::detail {
class InterpreterDelegateOpInterface;
class BytecodeDelegateOpInterface;

/// This class defines a delegate op interface to
/// `BytecodeInterpreterOpInterface` for operations that define a simple
/// `interpret` method with no additional bytecode payload. This is poor man's
/// interface inheritance. Most of the code here is boilerplate.
struct InterpreterDelegateOpInterfaceTraits
    : public BytecodeInterpreterOpInterfaceInterfaceTraits {
  template <typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = InterpreterDelegateOpInterface;
    Model() : Concept{getInterpretHook} {}

    /// This method defines the delegate `interpret` hook to call into the
    /// concrete operation's `interpret` method.
    static inline OpBytecodeGenerator getInterpretHook() {
      return {0, nullptr,
              +[](Operation *op, ArrayRef<Attribute> operands,
                  const void *payload, InterpreterState &state) {
                return cast<ConcreteOp>(op).interpret(operands, state);
              }};
    }
  };
};

struct BytecodeDelegateOpInterfaceTraits
    : public BytecodeInterpreterOpInterfaceInterfaceTraits {
  template <typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = InterpreterDelegateOpInterface;
    Model() : Concept{getInterpretHook} {}

    static inline OpBytecodeGenerator getInterpretHook() {
      using Payload = typename ConcreteOp::Payload;
      return {sizeof(Payload),
              +[](Operation *op, void *payload, TargetInfoAttr target) {
                return cast<ConcreteOp>(op).compile(*(Payload *)payload,
                                                    target);
              },
              +[](Operation *op, ArrayRef<Attribute> operands,
                  const void *payload, InterpreterState &state) {
                return cast<ConcreteOp>(op).interpret(
                    operands, *(const Payload *)payload, state);
              }};
    }
  };
};

template <typename ConcreteOp>
struct InterpreterDelegateOpInterfaceTrait;
template <typename ConcreteOp>
struct BytecodeDelegateOpInterfaceTrait;

class InterpreterDelegateOpInterface : public BytecodeInterpreterOpInterface {
public:
  template <typename ConcreteOp>
  struct Trait : public InterpreterDelegateOpInterfaceTrait<ConcreteOp> {};
};
class BytecodeDelegateOpInterface : public BytecodeInterpreterOpInterface {
public:
  template <typename ConcreteOp>
  struct Trait : public BytecodeDelegateOpInterfaceTrait<ConcreteOp> {};
};

template <typename ConcreteOp>
struct InterpreterDelegateOpInterfaceTrait
    : public mlir::OpInterface<
          InterpreterDelegateOpInterface,
          InterpreterDelegateOpInterfaceTraits>::Trait<ConcreteOp> {};
template <typename ConcreteOp>
struct BytecodeDelegateOpInterfaceTrait
    : public mlir::OpInterface<
          BytecodeDelegateOpInterface,
          BytecodeDelegateOpInterfaceTraits>::Trait<ConcreteOp> {};
} // namespace M::detail

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
