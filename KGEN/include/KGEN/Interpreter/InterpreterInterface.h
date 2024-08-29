//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
#define SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H

#include "Support/Compiler/ErrorTree.h"
#include "mlir/IR/OpDefinition.h"

namespace M {
class InterpreterState;

using InterpretHook = ErrorTreeOrSuccess (*)(Operation *, ArrayRef<Attribute>,
                                             const void *, InterpreterState &);
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
    static inline InterpretHook getInterpretHook() {
      return +[](Operation *op, ArrayRef<Attribute> operands,
                 const void *payload, InterpreterState &state) {
        return cast<ConcreteOp>(op).interpret(operands, state);
      };
    }
  };
};

template <typename ConcreteOp>
struct InterpreterDelegateOpInterfaceTrait;

class InterpreterDelegateOpInterface : public BytecodeInterpreterOpInterface {
public:
  template <typename ConcreteOp>
  struct Trait : public InterpreterDelegateOpInterfaceTrait<ConcreteOp> {};
};

template <typename ConcreteOp>
struct InterpreterDelegateOpInterfaceTrait
    : public mlir::OpInterface<
          InterpreterDelegateOpInterface,
          InterpreterDelegateOpInterfaceTraits>::Trait<ConcreteOp> {};
} // namespace M::detail

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
