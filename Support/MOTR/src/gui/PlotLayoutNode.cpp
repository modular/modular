//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "PlotLayoutNode.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>

#include "implot.h"

namespace M::motr::Gui {

// Forward declaration of the template function
template <typename T>
static LayoutNode::Ptr cloneLayoutNode(const T *srcNode);

PlotLayoutNode::PlotLayoutNode(std::shared_ptr<LayoutNode> parent)
    : LayoutNode(parent) {
  type = LayoutNodeType::Plot;

  // Generate default sine wave data if no data is provided
  xData.clear();
  yData.clear();
  for (int i = 0; i < 100; i++) {
    float x = i * 0.01f * 2.0f * 3.14159f;
    xData.push_back(x);
    yData.push_back(std::sin(x));
  }

  plotTitle = "Sine Wave";
  xLabel = "x";
  yLabel = "sin(x)";
}

void PlotLayoutNode::draw(DrawContext &context) {
  LayoutNode::setContextPosition(context, true);

  // Set up ImPlot flags
  int plotFlags = ImPlotFlags_None;
  if (!showLegend)
    plotFlags |= ImPlotFlags_NoLegend;

  // Create plot
  if (ImPlot::BeginPlot(plotTitle.c_str(),
                        ImVec2(context.width, context.height), plotFlags)) {
    // Set up axes
    if (!xLabel.empty())
      ImPlot::SetupAxis(ImAxis_X1, xLabel.c_str());
    if (!yLabel.empty())
      ImPlot::SetupAxis(ImAxis_Y1, yLabel.c_str());

    if (!showGrid) {
      ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoGridLines);
      ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoGridLines);
    }

    // Plot data based on plot type
    auto color = get_color();
    ImU32 plotColor = IM_COL32(color.r, color.g, color.b, color.a);
    ImPlot::PushStyleColor(ImPlotCol_Line, plotColor);

    switch (plotType) {
    case PlotType::Line:
      ImPlot::PlotLine("Data", xData.data(), yData.data(),
                       static_cast<int>(xData.size()));
      break;
    case PlotType::Scatter:
      ImPlot::PlotScatter("Data", xData.data(), yData.data(),
                          static_cast<int>(xData.size()));
      break;
    case PlotType::Bar:
      ImPlot::PlotBars("Data", xData.data(), yData.data(),
                       static_cast<int>(xData.size()), 0.5f);
      break;
    case PlotType::Histogram:
      ImPlot::PlotHistogram("Data", yData.data(),
                            static_cast<int>(yData.size()));
      break;
    case PlotType::Pie:
      if (!yData.empty()) {
        const char *labels[] = {"Slice 1", "Slice 2", "Slice 3", "Slice 4",
                                "Slice 5"};
        int labelCount = std::min(static_cast<int>(yData.size()), 5);
        ImPlot::PlotPieChart(labels, yData.data(), labelCount, 0.5, 0.5, 0.4,
                             "%.1f", 90);
      }
      break;
    }

    ImPlot::PopStyleColor();
    ImPlot::EndPlot();
  }
}

} // namespace M::motr::Gui
