//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Strongly-typed DiagID enum, one enumerator per entry in DiagnosticIDs.def.
// Misspelled or removed IDs are compile errors, not silent runtime misses.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_DIAGNOSTICS_DIAGNOSTICIDS_H
#define KGEN_DIAGNOSTICS_DIAGNOSTICIDS_H

namespace M::KGEN::Diag {

/// Strongly-typed enumeration of all registered KGEN diagnostics.
/// Integer value equals the index in DiagnosticIDs.def for O(1) registry
/// lookup.
enum class DiagID : unsigned {
#define DIAG(ID, Category, Subsystem, Message) ID,
#include "KGEN/Diagnostics/DiagnosticIDs.def"
  NumDiags ///< Sentinel – total number of registered diagnostics.
};

} // namespace M::KGEN::Diag

#endif // KGEN_DIAGNOSTICS_DIAGNOSTICIDS_H
