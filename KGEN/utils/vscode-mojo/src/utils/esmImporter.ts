//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

/**
 * This utility allows importing from ESM modules, which are otherwise non
 * importable by typescript.
 */
export const esmImporter =
    new Function("specifier", 'return import(specifier)');