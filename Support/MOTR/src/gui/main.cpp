//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GlobalState.h"
#include "RenderPasses/RenderPasses.h"
#include "RenderView.h"

#include <memory>
#include <vector>

int main(int, char **) {

  using namespace M::motr::Gui;
  GlobalState &state = globalState();

  state.renderWindow = std::make_unique<RenderWindow>(
      "#main-render-window-canvas", "#canvas-container");

  RenderWindow &renderWindow = *state.renderWindow;
  if (!renderWindow.window)
    return 1;

  state.initWebSockets();

  RenderView renderView(renderWindow);
  std::vector<std::shared_ptr<RenderViewPass>> passes = {
      renderView.addPass(preRenderPass),
      renderView.addPass(processEventTree),
      renderView.addPass(renderEventTreeTable),
      renderView.addPass(renderEventTreeView),
      renderView.addPass(renderImGuiCoreWidgets),
      renderView.addPass(renderWindowNodeLayouts),
      renderView.addPass(createRootLayout),
      renderView.addPass(showFooter),
      // renderView.addPass(showStringLibrary),
      renderView.addPass(postRenderPass),
  };

  renderView.executeEventLoop();

  return 0;
}
