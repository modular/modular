//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_PLOT_LAYOUT_NODE_H
#define M_MOTR_GUI_PLOT_LAYOUT_NODE_H

#include "LayoutNode.h"
#include <string>
#include <vector>

namespace M::motr::Gui {

enum class PlotType { Line, Scatter, Bar, Histogram, Pie };

struct PlotLayoutSchema {
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(color,         // name
                                 Color::RGBA32, // type
                                 (Color::RGBA32{255, 255, 255,
                                                255})); // default value
};

struct PlotLayoutNode : public LayoutNode, public PlotLayoutSchema {
  using Ptr = std::shared_ptr<PlotLayoutNode>;
  PlotLayoutNode(std::shared_ptr<LayoutNode> parent);
  void draw(DrawContext &context) override;
  LayoutNode::Ptr clone() const override;

  PlotType plotType = PlotType::Line;
  std::vector<float> xData;
  std::vector<float> yData;
  std::string xLabel;
  std::string yLabel;
  std::string plotTitle;
  bool showLegend = true;
  bool showGrid = true;
};

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_PLOT_LAYOUT_NODE_H
