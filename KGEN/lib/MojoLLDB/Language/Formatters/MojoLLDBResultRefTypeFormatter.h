//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLLDBRESULTREFTYPEFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLLDBRESULTREFTYPEFORMATTER_H

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-forward.h"

namespace M::KGEN::Mojo {
/// Synthetic type factory that handles a formatter for LIT::REPLResultRefType.
lldb_private::SyntheticChildrenFrontEnd *
mojoREPLResultRefTypeSyntheticFrontEndCreator(
    lldb_private::CXXSyntheticChildren *, const lldb::ValueObjectSP &valobjSP);

/// Summary provider that handles LIT::REPLResultRefType by delegating the
/// summary construction to its inner object.
bool mojoREPLResultRefTypeSummaryProvider(
    lldb_private::ValueObject &valobj, lldb_private::Stream &stream,
    const lldb_private::TypeSummaryOptions &summaryOptions);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLLDBRESULTREFTYPEFORMATTER_H
