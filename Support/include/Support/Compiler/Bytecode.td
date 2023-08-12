//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BYTECODE_TD
#define SUPPORT_BYTECODE_TD

include "mlir/IR/BytecodeBase.td"

def String :
  WithParser<"succeeded($_reader.readString($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedString($_getter)",
  WithType<"StringRef">>>>;

def TypedAttr :
  WithParser<"succeeded($_reader.readAttribute($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttribute($_getter)",
  WithType<"TypedAttr">>>>;

class ConcreteType<string type> :
  WithParser<"succeeded($_reader.readType($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeType($_getter)",
  WithType<type>>>>;

class ConcreteAttribute<string type> :
  WithParser<"succeeded($_reader.readAttribute($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttribute($_getter)",
  WithType<type>>>>;

class OptionalAttribute<string type> :
  WithParser<"succeeded($_reader.readOptionalAttribute($_var))",
  WithPrinter<"$_writer.writeOptionalAttribute($_getter)",
  WithType<type>>>;

class EnumAttr<string type> :
  WithParser<"succeeded(M::readEnum<" # type # ">($_reader, $_var))",
  WithPrinter<"$_writer.writeVarInt(static_cast<uint64_t>($_getter))",
  WithType<type>>>;

def StringAttrArray :
  WithParser<"succeeded($_reader.readAttributes($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttributes($_getter)",
  WithType<"SmallVector<StringAttr>">>>>;

def TypedAttrArray :
  WithParser<"succeeded($_reader.readAttributes($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttributes($_getter)",
  WithType<"SmallVector<TypedAttr>">>>>;

#endif // SUPPORT_BYTECODE_TD
