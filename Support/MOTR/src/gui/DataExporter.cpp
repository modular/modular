//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DataExporter.h"
#include "GlobalState.h"
#include "imgui.h"
#include "motr/MString.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace M::motr::Gui {

void triggerJavascriptDownloadText(const std::string_view text,
                                   const std::string_view basename,
                                   const std::string_view extension) {
  // clang-format off
  EM_ASM({
    const text = UTF8ToString($0);
    const basename = UTF8ToString($1);
    const extension = UTF8ToString($2);
    downloadText(text, basename, extension);
  }, text.data(), basename.data(), extension.data());
  // clang-format on
}

double timeNsToExcelDate(int64_t ts) {
  double val = ts / 86400.0 / 1000000000.0;
  return val + 25569.0;
}

EventTreeNode::Ptr getPopEvent(const EventTreeNode::Ptr &n) {
  for (auto &c : n->children) {
    if (c->message.flags == MessageFlags::Pop) {
      return c;
    }
  }
  return nullptr;
}

int64_t getDuration(const EventTreeNode::Ptr &pushEvent) {
  if (pushEvent->message.flags == MessageFlags::Push) {
    auto popEvent = getPopEvent(pushEvent);
    if (popEvent) {
      return popEvent->message.ts - pushEvent->message.ts;
    }
  }
  return 0;
}

template <typename T>
std::optional<typename T::key_type> getMaxKey(const T &mapContainer) {
  auto rfirst = mapContainer.rbegin();
  auto rlast = mapContainer.rend();
  if (rfirst == rlast) {
    return std::nullopt;
  }
  return rfirst->first;
}

Rows exportTreeToRows(const EventTreeNode::Ptr &root) {

  std::vector<std::string> colNames;
  std::unordered_map<std::string, size_t> colIdx;

  auto addCol = [&](const std::string &colName) -> size_t {
    if (colIdx.find(colName) == colIdx.end()) {
      colIdx[colName] = colNames.size();
      colNames.push_back(colName);
    }
    return colIdx[colName];
  };

  MString sourceLocKey{Constants::source_loc::sv};
  MString processIdKey{Constants::ProcessId::sv};
  MString threadIdKey{Constants::ThreadId::sv};
  MString nameKey{Constants::name::sv};
  MString traceNameKey{Constants::TraceName::sv};
  MString sourceFileKey{Constants::SourceFile::sv};
  MString sourceLineKey{Constants::SourceLine::sv};
  MString programNameKey{Constants::ProgramName::sv};
  constexpr std::string_view unknown{"<unknown>"};

  constexpr bool emitTags = false;

  addCol("idx");
  addCol("pidx");
  addCol("depth");
  addCol(nameKey.str());
  addCol("ts");
  addCol("dur_ns");
  addCol("ts_iso8601");
  addCol("ts_excel");
  // addCol("type");
  addCol("id");
  if (emitTags) {
    addCol("key");
    addCol("val");
  }
  addCol(sourceLocKey.str());
  addCol(processIdKey.str());
  addCol(threadIdKey.str());

  struct RowCol {
    size_t row;
    size_t col;
    bool operator<(const RowCol &other) const {
      return row < other.row || (row == other.row && col < other.col);
    }
  };

  std::map<size_t, std::map<size_t, std::string>> values;

  auto setValue = [&](size_t rowIdx, std::string_view colName,
                      const std::string &value) -> bool {
    size_t colIdx = addCol(std::string{colName});

    // special case for the root node row
    if (rowIdx == 1) {
      if (colName != "idx" && colName != "pidx" && colName != "depth" &&
          colName != "name")
        return false;

      if (colName == "name") {
        values[rowIdx][colIdx] = "root";
        return true;
      }
    }
    if (value.empty())
      return false;
    values[rowIdx][colIdx] = value;
    return true;
  };

  std::unordered_set<Hash::Value> seenKeys;
  seenKeys.insert(nameKey.hash);
  seenKeys.insert(processIdKey.hash);
  seenKeys.insert(threadIdKey.hash);
  seenKeys.insert(traceNameKey.hash);
  seenKeys.insert(sourceLocKey.hash);
  seenKeys.insert(sourceFileKey.hash);
  seenKeys.insert(sourceLineKey.hash);

  std::unordered_map<uint64_t, int> idToRowIdx;
  int rowIdx = 0;
  auto dfs = [&](auto &&self, const EventTreeNode::Ptr &n, int depth) -> void {
    if (!emitTags) {
      if (!n || n->message.flags != MessageFlags::Push)
        return;
    } else {
      if (!n || n->message.flags == MessageFlags::Pop)
        return;
    }
    rowIdx++;
    const Message &m = n->message;
    int parent = idToRowIdx[m.pid];
    idToRowIdx[m.id] = rowIdx;

    std::string key, val;
    switch (m.flags) {
    case MessageFlags::TagStr:
      key = MString{m.id}.str();
      val = MString{m.ts}.str();
      break;
    case MessageFlags::TagInt:
      key = MString{m.id}.str();
      val = fmt::format("{}", m.ts);
      break;
    default:
      break;
    }
    int64_t dur = getDuration(n);
    setValue(rowIdx, "idx", fmt::format("{}", rowIdx));
    setValue(rowIdx, "pidx", fmt::format("{}", parent));
    setValue(rowIdx, "depth", fmt::format("{}", depth));
    setValue(rowIdx, "ts", fmt::format("{}", m.ts));
    setValue(rowIdx, "dur_ns", fmt::format("{}", dur));
    setValue(rowIdx, "ts_iso8601", fmt::format("{}", timeNsToISODate(m.ts)));
    setValue(rowIdx, "ts_excel", fmt::format("{}", timeNsToExcelDate(m.ts)));
    setValue(rowIdx, "type", fmt::format("{}", toString(m.type)));
    setValue(rowIdx, "id", fmt::format("0x{:016x}", m.id));
    if (emitTags) {
      setValue(rowIdx, "key", fmt::format("\"{}\"", key));
      setValue(rowIdx, "val", fmt::format("\"{}\"", val));
    }

    TagLibrary::Ptr tagLibrary = n->getTagLibrary();
    bool localOnly = tagLibrary->isLocalOnly();
    tagLibrary->setLocalOnly(false);

    uint64_t processID = tagLibrary->getOptionalU64(processIdKey).value_or(0);
    uint64_t threadID = tagLibrary->getOptionalU64(threadIdKey).value_or(0);

    if (m.type == MessageType::Process) {
      tagLibrary->setString(nameKey, fmt::format("process {}", processID));
    } else if (m.type == MessageType::Thread) {
      tagLibrary->setString(
          nameKey, fmt::format("process {} thread {}", processID, threadID));
    } else {
      std::string traceName{
          tagLibrary->getOptionalString(traceNameKey).value_or(unknown)};
      tagLibrary->setString(nameKey, traceName);
    }

    if (tagLibrary->hasTag(programNameKey)) {
      std::string programName{tagLibrary->getString(programNameKey)};
      std::string name{tagLibrary->getString(nameKey)};
      std::string val = fmt::format("{} {}", programName, name);
      tagLibrary->setString(nameKey, val);
    }

    // source location
    {
      std::string sourceLoc{tagLibrary->getString(sourceFileKey)};
      if (!sourceLoc.empty()) {
        uint64_t sourceLine = tagLibrary->getU64(sourceLineKey);
        if (sourceLine > 0) {
          sourceLoc = fmt::format("{}:{}", sourceLoc, sourceLine);
        }
        tagLibrary->setString(sourceLocKey, sourceLoc);
      }
    }
    setValue(
        rowIdx, sourceLocKey.sv(),
        std::string{tagLibrary->getOptionalString(sourceLocKey).value_or("")});
    setValue(rowIdx, processIdKey.sv(), fmt::format("{}", processID));
    setValue(rowIdx, threadIdKey.sv(), fmt::format("{}", threadID));
    setValue(rowIdx, nameKey.sv(),
             std::string{
                 tagLibrary->getOptionalString(nameKey).value_or("<unknown>")});

    if (m.flags == MessageFlags::Push) {
      tagLibrary->setLocalOnly(true);

      std::set<std::string> tagValues;
      for (auto &[key, _] : tagLibrary->tagStrMap) {
        if (seenKeys.find(key.hash) != seenKeys.end())
          continue;
        auto value = fmt::format("{}={}", key.sv(), tagLibrary->getString(key));
        tagValues.insert(value);
      }

      for (auto &[key, _] : tagLibrary->tagIntMap) {
        if (seenKeys.find(key.hash) != seenKeys.end())
          continue;
        auto value = tagLibrary->getU64(key);
        auto str = fmt::format("{}={} (0x{:x})", key.sv(), value, value);
        tagValues.insert(str);
      }

      std::string tagValuesStr = fmt::format("{}", fmt::join(tagValues, ", "));
      setValue(rowIdx, "tags", tagValuesStr);
    }

    tagLibrary->setLocalOnly(localOnly);

    for (auto &c : n->children)
      self(self, c, depth + 1);
  };
  dfs(dfs, root, 0);

  // now the sparse values map is populated
  // we need to convert it to a dense matrix

  // set the header row
  for (auto &colName : colNames) {
    setValue(0, colName, colName);
  }

  // determine the size of the dense matrix
  size_t maxRowIdx = getMaxKey(values).value_or(0);
  size_t maxColIdx = 0;
  for (auto &[rowIdx, rowValues] : values) {
    size_t numCols = getMaxKey(rowValues).value_or(0);
    maxColIdx = numCols > maxColIdx ? numCols : maxColIdx;
  }

  // convenience function to get a value from the sparse map
  auto getValue = [&](size_t rowIdx, size_t colIdx) -> std::string_view {
    auto rit = values.find(rowIdx);
    if (rit == values.end())
      return {};
    auto &rowValues = rit->second;
    auto cit = rowValues.find(colIdx);
    if (cit == rowValues.end())
      return {};
    return cit->second;
  };

  Rows rows;
  rows.reserve(maxRowIdx + 1);
  for (size_t rowIdx = 0; rowIdx <= maxRowIdx; ++rowIdx) {
    std::vector<std::string> row;
    row.reserve(maxColIdx + 1);
    for (size_t colIdx = 0; colIdx < colNames.size(); ++colIdx) {
      row.push_back(std::string{getValue(rowIdx, colIdx)});
    }
    rows.push_back(row);
  }

  return rows;
}
std::string rowsToText(const Rows &rows, std::string_view separator) {
  const std::string sep{separator};
  const std::string newLine = "\n";

  const size_t sepSize = sep.size();
  const size_t newLineSize = newLine.size();

  size_t totalChars = 0;
  for (auto &row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      totalChars += row[i].size();
      if (i < row.size() - 1) {
        totalChars += sepSize;
      }
    }
    totalChars += newLineSize;
  }

  std::string text;
  text.reserve(totalChars);

  for (auto &row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      text += row[i];
      if (i < row.size() - 1) {
        text += sep;
      }
    }
    text += newLine;
  }
  MOTR_LOG("text.size() = {}, totalChars = {}", text.size(), totalChars);
  // assert(text.size() == totalChars);

  return text;
}

std::string exportTreeToText(const EventTreeNode::Ptr &root,
                             std::string_view separator) {
  Rows rows = exportTreeToRows(root);
  std::string sep = separator.empty() ? "," : std::string{separator};
  return rowsToText(rows, sep);
}

std::string makeTSV(const Rows &rows) {
  std::string out;
  for (auto &r : rows) {
    for (size_t i = 0; i < r.size(); ++i) {
      out += r[i];
      if (i + 1 < r.size())
        out += '\t';
    }
    out += '\n';
  }
  return out;
}

std::string makeHTML(const Rows &rows) {
  std::string h = "<table><thead><tr>";
  for (auto &cell : rows.front())
    h += "<th>" + cell + "</th>";
  h += "</tr></thead><tbody>";
  for (size_t r = 1; r < rows.size(); ++r) {
    h += "<tr>";
    for (auto &cell : rows[r])
      h += "<td>" + cell + "</td>";
    h += "</tr>";
  }
  h += "</tbody></table>";
  return h;
}

void copyTableToClipboard(const std::string &tsv, const std::string &html) {
#ifdef __EMSCRIPTEN__
  // clang-format off
  EM_ASM({
    const tsv   = UTF8ToString($0);
    const html  = UTF8ToString($1);

    (async () => {
      try {
        const data = {
          "text/plain": new Blob([tsv ], {type: "text/plain"}),
          "text/html" : new Blob([html], {type: "text/html"})
        };
        await navigator.clipboard.write([new ClipboardItem(data)]);
        console.log("copied!");
      } catch(e) {
        console.error("clipboard error", e);
      }
    })();
  }, tsv.c_str(), html.c_str());
// clang-format on
#endif
}

} // namespace M::motr::Gui
