//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_DATA_EXPORTER_H
#define M_MOTR_GUI_DATA_EXPORTER_H

#include "motr/EventTree.h"
#include <string>
#include <vector>

namespace M::motr::Gui {

void triggerJavascriptDownloadText(const std::string_view text,
                                   const std::string_view basename,
                                   const std::string_view extension);

using Rows = std::vector<std::vector<std::string>>;

std::string exportTreeToText(const EventTreeNode::Ptr &root,
                             std::string_view separator);

void copyTableToClipboard(const std::string &tsv, const std::string &html);
std::string makeTSV(const Rows &rows);
std::string makeHTML(const Rows &rows);
std::string getBasename(const EventTreeNode::Ptr &node);

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_DATA_EXPORTER_H
