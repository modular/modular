//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_RENDER_PASSES_H
#define M_MOTR_GUI_RENDER_PASSES_H

#include "RenderPasses/RenderEventTreeView.h" // for renderEventTreeView

namespace M::motr::Gui {

struct RenderView;

// Core rendering passes
void preRenderPass(RenderView &renderView);
void postRenderPass(RenderView &renderView);
void createDockSpace(RenderView &renderView);
// UI components
void renderImGuiCoreWidgets(RenderView &renderView);
void showDebugBanner(RenderView &renderView);
void showFooter(RenderView &renderView);
void showStringLibrary(RenderView &renderView);

// Event tree rendering
void renderEventTreeTable(RenderView &renderView);
void renderProcTrees(RenderView &renderView);
void renderEventProcessWindows(RenderView &renderView);

// Layout rendering
void renderWindowNodeLayouts(RenderView &renderView);
void createRootLayout(RenderView &renderView);

// Flame graph rendering
void createFlameWindows(RenderView &renderView);
void renderFlameWindows(RenderView &renderView);

void processEventTree(RenderView &renderView);

} // namespace M::motr::Gui
#endif // M_MOTR_GUI_RENDER_PASSES_H
