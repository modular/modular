//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROPERTIES_TD
#define SUPPORT_PROPERTIES_TD

include "mlir/IR/Properties.td"

class EnumProperty<string storageTypeParam, string desc = ""> :
    Property<storageTypeParam, desc> {
  code writeToMlirBytecode = [{
    $_writer.writeVarInt(static_cast<uint64_t>($_storage));
  }];
  code readFromMlirBytecode = [{
    uint64_t val;
    if (failed($_reader.readVarInt(val)))
      return ::mlir::failure();
    $_storage = static_cast<}] # storageTypeParam # [{>(val);
  }];
}

#endif // SUPPORT_PROPERTIES_TD
