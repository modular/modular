//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RenderEventTreeView.h"
#include "DataExporter.h"
#include "DrawEventTree.h"
#include "GlobalState.h"
#include "WebUtilities.h"
#include "imgui.h"

namespace M::motr::Gui {

void renderEventTreeView(RenderView &) {
  ImGui::Begin("Event Tree View###EventTreeView", nullptr,
               ImGuiWindowFlags_NoCollapse);
  ImGui::SetWindowSize(ImVec2(1400, 600), ImGuiCond_Once);

  auto &st = treeState();
  auto &tree = globalState().getEventTree();

  // Layout: Filter: [input field........................] [Clear] [Export TSV]

  // Calculate button widths
  float clearButtonWidth = ImGui::CalcTextSize("Clear Filter").x +
                           ImGui::GetStyle().FramePadding.x * 2;
  float exportButtonWidth = ImGui::CalcTextSize("Export TSV").x +
                            ImGui::GetStyle().FramePadding.x * 2;
  float filterLabelWidth = ImGui::CalcTextSize("Filter:").x;
  float spacing = ImGui::GetStyle().ItemSpacing.x;

  // Calculate input width (fill remaining space)
  float availableWidth = ImGui::GetContentRegionAvail().x;
  float inputWidth = availableWidth - filterLabelWidth - clearButtonWidth -
                     exportButtonWidth - (spacing * 3);

  // Filter label
  ImGui::Text("Filter:");
  ImGui::SameLine();

  // Input field (expanded to fill space)
  ImGui::PushItemWidth(inputWidth);
  ImGui::PushID("EventTreeFilter"); // Add unique ID to prevent conflicts
  bool filterChanged = ImGui::InputText("", st.filter, sizeof(st.filter));
  ImGui::PopID();
  ImGui::PopItemWidth();
  ImGui::SameLine();

  // Clear button
  if (ImGui::Button("Clear Filter")) {
    st.filter[0] = '\0';
    filterChanged = true;
  }
  ImGui::SameLine();

  // Export TSV button (right-aligned)
  if (ImGui::Button("Export TSV")) {
    auto text = exportTreeToText(tree.root, "\t");
    auto basename = getBasename(tree.root);
    triggerJavascriptDownloadText(text, basename, "tsv");
    ImGui::SetClipboardText(text.c_str());
  }

  // If filter changed, force rebuild of flattened view
  if (filterChanged) {
    st.lastTreeGeneration = -1;
  }

  ImGui::Separator();

  st.row = 0;

  const ImGuiTableFlags tblFlags =
      ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV;

  // Calculate available space for table (leave room for status bar)
  float statusBarHeight = ImGui::GetTextLineHeightWithSpacing() +
                          ImGui::GetStyle().ItemSpacing.y * 2;
  float availableHeight = ImGui::GetContentRegionAvail().y - statusBarHeight;

  // Push unique ID for the table to prevent input conflicts
  ImGui::PushID("EventTreeTable");
  // Updated column order: name, age, duration, date, time, tags, source_loc
  if (ImGui::BeginTable("Table", 7, tblFlags, ImVec2(0, availableHeight))) {
    ImGui::TableSetupScrollFreeze(1, 0);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableSetupColumn("Age (ms)", ImGuiTableColumnFlags_WidthFixed, 80);
    ImGui::TableSetupColumn("Duration (ms)", ImGuiTableColumnFlags_WidthFixed,
                            100);
    ImGui::TableSetupColumn("Date (GMT)", ImGuiTableColumnFlags_WidthFixed,
                            100);
    ImGui::TableSetupColumn("Time (GMT)", ImGuiTableColumnFlags_WidthFixed,
                            160);
    ImGui::TableSetupColumn("Tags", ImGuiTableColumnFlags_WidthFixed, 200);
    ImGui::TableSetupColumn("Source Location",
                            ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableHeadersRow();

    drawEventTreeNodes(tree.root->children);

    ImGui::EndTable();
  }
  ImGui::PopID();

  // Status bar at the bottom
  ImGui::Separator();
  int totalRenderedEvents = static_cast<int>(st.flattenedNodes.size());
  ImGui::Text("Events: %d", totalRenderedEvents);

  ImGui::End();
}

} // namespace M::motr::Gui
