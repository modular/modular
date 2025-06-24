//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RenderPasses.h"
#include "DataExporter.h"
#include "EventTreeProcessor.h"
#include "FlameGraphRenderer.h"
#include "GlobalState.h"
#include "LayoutLoader.h"
#include "RenderView.h"
#include "motr/RPCMailbox.h"

#include "imgui.h"
#include "motr/Hash.h"
#include "motr/Log.h"
#include "motr/Time.h"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <algorithm>
#include <thread>
#include <unordered_set>

#include "motr/RPCMailbox.h"
#include "motr/RPCMethods.h"
#include "motr/Types/Types.h"

namespace M::motr::Gui {

// Forward declarations for functions defined later
std::string getChangingColor();
void updateTagLibrary();
std::string getBuildInfo();
double getAverageFramerate();
std::string getElapsedTimeString();

auto splitDateTime =
    [](const std::string &datetimestr) -> std::pair<std::string, std::string> {
  if (datetimestr.size() == 27)
    return {datetimestr.substr(0, 10), datetimestr.substr(11, 11)};
  return {datetimestr, ""};
};

void renderEventTreeTable(RenderView &renderView) {
  static const ImGuiTableFlags flags =
      ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
      ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable |
      ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg |
      ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;

  ImGui::Begin("Event Tree###EventTree", nullptr, ImGuiWindowFlags_NoCollapse);
  ImGui::SetWindowSize(ImVec2(1000, 500), ImGuiCond_Once);

  auto &stringLibrary = globalState().getStringLibrary();
  auto now = Time::now();

  if (ImGui::BeginTable("EventTreeTable", 10, flags)) {
    ImGui::TableSetupColumn("Date", ImGuiTableColumnFlags_WidthFixed, 120);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 120);
    ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_WidthFixed, 80);
    ImGui::TableSetupColumn("Age", ImGuiTableColumnFlags_None, 80);
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_None, 200);
    ImGui::TableSetupColumn("PID", ImGuiTableColumnFlags_None, 200);
    ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_None, 100);
    ImGui::TableSetupColumn("Flags", ImGuiTableColumnFlags_None, 100);
    ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_None);
    ImGui::TableSetupColumn("Val", ImGuiTableColumnFlags_None);
    ImGui::TableHeadersRow();

    auto &allNodes = globalState().getFlatEventTree();

    // Use ImGuiListClipper for virtual scrolling
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(allNodes.size()));

    while (clipper.Step()) {
      for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
        const auto &node = allNodes[i];
        ImGui::TableNextRow();

        auto &msg = node->message;

        std::string id = fmt::format("{:016x}", msg.id);
        std::string pid = fmt::format("{:016x}", msg.pid);
        std::string type = toString(msg.type);
        std::string flags = toString(msg.flags);
        std::string key;
        std::string val;

        std::string datetimestr;
        std::string startstr;
        std::string age;
        if (msg.flags != MessageFlags::TagStr &&
            msg.flags != MessageFlags::TagInt) {
          datetimestr = timeNsToISODate(msg.ts);
          startstr = Time::getElapsedTimeString(msg.ts);
          age = (now - Time::Timestamp(msg.ts)).toString();
        } else {
          datetimestr = "";
          startstr = "";
          age = "";
          key = stringLibrary.getString(msg.id, true);
          if (msg.flags == MessageFlags::TagInt)
            val = fmt::format("{}", msg.ts);
          else
            val = stringLibrary.getString(msg.ts, true);
        }

        auto [datestr, timestr] = splitDateTime(datetimestr);

        if (ImGui::TableNextColumn())
          ImGui::Text("%s", datestr.c_str());
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", timestr.c_str());
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", startstr.c_str());
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", age.c_str());
        if (ImGui::TableNextColumn()) {
          ImGui::Text("%s", id.c_str());
          if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("ID: %s", id.c_str());
            ImGui::EndTooltip();
          }
        }
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", pid.c_str());
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", type.c_str());
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", flags.c_str());
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", key.c_str());
        if (ImGui::TableNextColumn())
          ImGui::Text("%s", val.c_str());
      }
    }

    ImGui::EndTable();
  }
  ImGui::End();
}

void renderWindowNodeLayouts(RenderView &renderView) {
  auto &state = globalState();
  auto &tagLibrary = state.getTagLibrary();
  LayoutNode::DrawContext context{&tagLibrary};

  tagLibrary.setString({"changing_color"}, getChangingColor());

  static std::vector<std::string_view> flexDir = {
      "row", "column", "row-reverse", "column-reverse"};
  static int crazy_index = 0;
  tagLibrary.setString({"flexDir"},
                       flexDir[(crazy_index / 10) % flexDir.size()]);
  crazy_index++;

  updateTagLibrary();

  for (auto &[name, layoutWindow] : state.layoutWindows) {
    if (layoutWindow->visible)
      layoutWindow->draw(context);
  }
}

void createRootLayout(RenderView &renderView) {
  constexpr std::string_view rootLayoutName = "RootView";
  auto &state = globalState();
  auto &layoutLibrary = state.getLayoutLibrary();

  auto libraryRootLayout = layoutLibrary.getLayout(std::string(rootLayoutName));
  if (!libraryRootLayout)
    return;

  auto &eventTree = state.getEventTree();
  auto &eventTreeRoot = eventTree.root;
  auto &eventTreeRootChildren = eventTreeRoot->children;
  for (auto &child : eventTreeRootChildren) {
    // Process nodes as needed
  }
}

void showFooter(RenderView &renderView) {
  auto &io = ImGui::GetIO();
  auto window_height = io.DisplaySize.y;
  auto window_width = io.DisplaySize.x;
  auto line_height = ImGui::GetTextLineHeight();
  auto footer_margin = 5;
  auto footer_height = line_height + footer_margin * 2;
  auto footer_y0 = window_height - footer_height;

  ImGui::SetNextWindowPos(ImVec2(0, footer_y0));
  ImGui::SetNextWindowSize(ImVec2(window_width, footer_height));

  auto grey = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
  auto black = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);

  ImGui::PushStyleColor(ImGuiCol_WindowBg, grey);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

  ImGui::Begin("##Footer", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoNav |
                   ImGuiWindowFlags_NoFocusOnAppearing);

  if (true) {
    ImGui::SetCursorPosX(footer_margin);
    ImGui::SetCursorPosY(footer_margin);
    std::string text = getBuildInfo();
    ImGui::TextColored(black, "%s", text.c_str());
  }

  int demo_offset = 30;

  if (true) {
    double framerate = getAverageFramerate();
    double frametime = 1000.0 / framerate;
    std::string text =
        fmt::format("{:.0f} FPS / {:0.2f}ms", framerate, frametime);
    auto text_width = ImGui::CalcTextSize(text.c_str()).x;
    auto text_x0 = window_width - text_width - footer_margin * 5 - demo_offset;
    ImGui::SetCursorPosX(text_x0);
    ImGui::SetCursorPosY(footer_margin);
    ImGui::TextColored(black, "%s", text.c_str());
  }

  auto &state = globalState();

  if (true) {
    ImGui::SetCursorPosX(window_width / 2);
    ImGui::SetCursorPosY(footer_margin);
    if (false && ImGui::Button("HostInfo")) {
      auto hostInfo = RPC::Methods::getHostInfo::Call();
      auto motrServerInfo = RPC::Methods::getMotrServerInfo::Call();

      // for (auto &websocket : state.websockets) {
      //   websocket.sendText(rpcCall.requestId.toString());
      // }
    }
  }
  if (true) {
    ImGui::SetCursorPosX(window_width - demo_offset);
    ImGui::SetCursorPosY(0);
    auto &show_demo_window = GlobalState::getSingleton().show_demo_window;
    ImGui::Checkbox("##DemoWindowCheckbox", &show_demo_window);
  }

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
}

void renderEventTreeRoot(RenderView &renderView, EventTreeNode::Ptr tree) {
  auto &state = globalState();

  std::string title = fmt::format("Process: {}", tree->message.procid);
  static std::unordered_map<std::string, int> counts;
  static bool isOpen[512];
  static bool once = false;

  if (!once) {
    std::fill(isOpen, isOpen + 512, true);
    once = true;
  }

  int &count = counts[title];
  if (count == 0) {
    count = counts.size();
    size_t off = count * 50;
    ImGui::SetWindowPos(ImVec2(off, off), ImGuiCond_Once);
  }

  if (!isOpen[count])
    return;

  static const ImGuiTableFlags flags =
      ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
      ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable |
      ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg |
      ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;

  ImGui::Begin(title.c_str(), isOpen + count, ImGuiWindowFlags_None);
  ImGui::SetWindowSize(ImVec2(800, 500), ImGuiCond_Once);

  auto &stringLibrary = globalState().getStringLibrary();
  auto now = Time::Timestamp::now();
  auto hightlight_color = ImVec4(0.4, 0.7, 1.0, 1.0f);
  auto highlight_id = state.highlightId;
  state.highlightId = 0;

  if (ImGui::BeginTable("EventTreeTable", 10, flags)) {
    float charWidth = ImGui::GetFontSize();
    int w = 10;
    ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 12 * w);
    ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, 15 * w);
    ImGui::TableSetupColumn("Val", ImGuiTableColumnFlags_WidthFixed, 15 * w);
    ImGui::TableSetupColumn("Date", ImGuiTableColumnFlags_WidthFixed, 5 * w);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 5 * w);
    ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_WidthFixed, 5 * w);
    ImGui::TableSetupColumn("Age", ImGuiTableColumnFlags_None, 5 * w);
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_None, 5 * w);
    ImGui::TableSetupColumn("PID", ImGuiTableColumnFlags_None, 5 * w);
    ImGui::TableSetupColumn("Flags", ImGuiTableColumnFlags_None, 8 * w);
    ImGui::TableHeadersRow();

    int row = 0;
    for (const auto &node : tree->getDFSPreOrder()) {
      auto &msg = node->message;

      if (node->message.flags == MessageFlags::TagStr ||
          node->message.flags == MessageFlags::TagInt) {
        // continue;
      }

      std::string durstr;
      std::string tagstr;
      std::string name;

      if (node->message.flags == MessageFlags::Push) {
        auto children = node->getChildrenWith(MessageFlags::Pop);
        for (auto &child : children) {
          int64_t dur = child->message.ts - msg.ts;
          auto dur_ms = dur / 1000000;
          durstr += fmt::format("{}ms ", dur_ms);
        }

        children = node->getChildrenWith(MessageFlags::TagStr);
        for (auto &child : children) {
          auto key = stringLibrary[child->message.id];
          auto val = stringLibrary[child->message.ts];
          tagstr += fmt::format(" [{}={}]", key, val);
          constexpr const Hash::Value TraceName{"TraceName"};
          if (TraceName == child->message.id) {
            name = val;
          }
        }

        children = node->getChildrenWith(MessageFlags::TagInt);
        for (auto &child : children) {
          auto key = stringLibrary[child->message.id];
          auto val = fmt::format("{}", child->message.ts);
          tagstr += fmt::format(" [{}={}]", key, val);
        }
      }

      std::string indent = std::string(node->numAncestors(), '.');
      std::string id = fmt::format("{:016x}", msg.id);
      std::string pid = fmt::format("{:016x}", msg.pid);
      std::string type = toString(msg.type);
      type = type == "Set" ? "Tag" : type;
      type = name.empty() ? type : name;
      type = indent + type;
      std::string flags = toString(msg.flags);
      std::string key;
      std::string val;

      std::string datetimestr;
      std::string startstr;
      std::string age;
      if (msg.flags != MessageFlags::TagStr &&
          msg.flags != MessageFlags::TagInt) {
        datetimestr = timeNsToISODate(msg.ts);
        startstr = Time::getElapsedTimeString(msg.ts);
        age = (now - Time::Timestamp(msg.ts)).toString();
      } else {
        datetimestr = "";
        startstr = "";
        age = "";
        key = stringLibrary.getString(msg.id, true);
        if (msg.flags == MessageFlags::TagInt)
          val = fmt::format("{}", msg.ts);
        else
          val = stringLibrary.getString(msg.ts, true);
      }

      auto [datestr, timestr] = splitDateTime(datetimestr);
      type = type + " " + durstr + " " + tagstr;

      ImGui::TableNextRow();

      if (ImGui::TableNextColumn())
        ImGui::Text("%s", type.c_str());
      if (ImGui::TableNextColumn())
        ImGui::Text("%s", key.c_str());
      if (ImGui::TableNextColumn())
        ImGui::Text("%s", val.c_str());
      if (ImGui::TableNextColumn())
        ImGui::Text("%s", datestr.c_str());
      if (ImGui::TableNextColumn())
        ImGui::Text("%s", timestr.c_str());
      if (ImGui::TableNextColumn())
        ImGui::Text("%s", startstr.c_str());
      if (ImGui::TableNextColumn())
        ImGui::Text("%s", age.c_str());

      if (ImGui::TableNextColumn()) {
        if (highlight_id == msg.id) {
          ImGui::TextColored(hightlight_color, "%s", id.c_str());
        } else {
          ImGui::Text("%s", id.c_str());
        }
        if (ImGui::IsItemHovered()) {
          state.highlightId = msg.id;
          if (msg.isaTag()) {
            ImGui::SetTooltip("Key = %s, Value = %s", key.c_str(), val.c_str());
          }
        }
      }

      if (ImGui::TableNextColumn()) {
        if (highlight_id == msg.pid) {
          ImGui::TextColored(hightlight_color, "%s", pid.c_str());
        } else {
          ImGui::Text("%s", pid.c_str());
        }
        if (ImGui::IsItemHovered()) {
          state.highlightId = msg.pid;
        }
      }

      if (ImGui::TableNextColumn())
        ImGui::Text("%s", flags.c_str());
      row++;
    }

    ImGui::EndTable();
  }
  ImGui::End();
}

void renderProcTrees(RenderView &renderView) {
  auto &eventTree = globalState().getEventTree();
  for (auto &node : eventTree.root->children) {
    renderEventTreeRoot(renderView, node);
  }
}

void renderImGuiCoreWidgets(RenderView &renderView) {
  ImGuiIO &io = ImGui::GetIO();
  static bool show_style_editor = false;

  bool &show_demo_window = GlobalState::getSingleton().show_demo_window;

  if (show_demo_window)
    ImGui::ShowDemoWindow(&show_demo_window);

  if (show_style_editor) {
    ImGui::Begin("Style Editor");
    ImGui::ShowStyleEditor();
    ImGui::End();
  }

  if (false) {
    ImGui::Begin("MOTR");
    ImGui::Checkbox("ImGuiDemo", &show_demo_window);
    ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate,
                io.Framerate);
    ImGui::End();
  }
}

void showStringLibrary(RenderView &renderView) {
  ImGui::Begin("String Library");
  auto &stringLibrary = globalState().getStringLibrary();
  ImGui::Text("String Library");
  ImGui::Text("Chunks: [%zu/%zu]", stringLibrary.chunks.size(),
              stringLibrary.maxChunks);
  ImGui::Text("Num Strings: %zu", stringLibrary.strings.size());
  int idx = 0;
  for (auto &[hash, sv] : stringLibrary.strings) {
    std::string msg = fmt::format("{:4d} 0x{:016x} [{:3d}]: {}", idx++, hash,
                                  sv.size(), shortString(sv, 30));
    ImGui::Text("%s", msg.c_str());
  }
  ImGui::End();
}

void showDebugBanner(RenderView &renderView) {
  static bool show_debug_banner = true;
  if (!show_debug_banner)
    return;

#ifdef NDEBUG
  const char *banner_text = "MOTR";
#else
  const char *banner_text = "MOTR Debug Build";
#endif

  auto text_size = ImGui::CalcTextSize(banner_text);
  auto window_margin = 10;
  auto window_wid = ImGui::GetIO().DisplaySize.x;
  auto window_hei = text_size.y + window_margin * 2;
  auto window_size = ImVec2(window_wid, window_hei);

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(window_size);

  auto background_color = ImVec4(0.3f, 0.3f, 0.5f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, background_color);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

  ImGui::Begin("##DebugBanner", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoNav |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);

  auto text_x0 = (window_size.x - text_size.x) * 0.5f;
  auto text_y0 = (window_size.y - text_size.y) * 0.5f;
  ImGui::SetCursorPosX(text_x0);
  ImGui::SetCursorPosY(text_y0);

  auto color_white = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImGui::TextColored(color_white, "%s", banner_text);

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
}

std::string getElapsedTimeString() {
  uint64_t elapsed_seconds = getElapsedSecondsSinceBuild();
  return secondsToTimeString(elapsed_seconds);
}

double getAverageFramerate() {
  ImGuiIO &io = ImGui::GetIO();
  constexpr size_t history_size = 256;
  static std::vector<double> framerates(history_size);
  static size_t index = 0;
  framerates[index % history_size] = io.Framerate;
  index++;
  double sum = 0;
  if (index < history_size) {
    for (size_t i = 0; i < index; i++) {
      sum += framerates[i];
    }
    sum = sum / index;
  } else {
    for (size_t i = 0; i < history_size; i++) {
      sum += framerates[i];
    }
    sum = sum / history_size;
  }
  return sum;
}

std::string getBuildInfo() {
#ifdef NDEBUG
  std::string prefix = "Build";
#else
  std::string prefix = "Debug Build";
#endif

  std::string text = fmt::format("{} {} at {} ({} ago)", prefix, __DATE__,
                                 __TIME__, getElapsedTimeString());
  return text;
}

void updateTagLibrary() {
  auto &state = globalState();
  auto &tagLibrary = state.getTagLibrary();
  tagLibrary.setString({"BuildInfo"}, getBuildInfo());

  double framerate = getAverageFramerate();
  double frametime = 1000.0 / framerate;
  tagLibrary.setString({"FrameRate"}, fmt::format("{:.0f}", framerate));
  tagLibrary.setString({"FrameTime"}, fmt::format("{:.2f}", frametime));
  tagLibrary.setString({"ElapsedSeconds"},
                       state.elapsed().toString(Time::Precision::Seconds));
  tagLibrary.setU64({"FrameCount"}, ImGui::GetFrameCount());
}

std::string getChangingColor() {
  using namespace Color;
  static HSVA hsva = {0.0f, 1.0f, 1.0f, 1.0f};
  hsva.h = (hsva.h + 0.1f);
  if (hsva.h > 360.0f)
    hsva.h = 0.0f;
  RGBA color = HSVAtoRGBA(hsva);
  RGBA32 color32 = {static_cast<uint8_t>(color.r * 255.0f),
                    static_cast<uint8_t>(color.g * 255.0f),
                    static_cast<uint8_t>(color.b * 255.0f),
                    static_cast<uint8_t>(color.a * 255.0f)};
  return fmt::format("#{:02x}{:02x}{:02x}{:02x}", color32.r, color32.g,
                     color32.b, color32.a);
}

void displayNodeAndChildren(const EventTreeNode::Ptr &node) {
  int indent = node->numAncestors();
  std::string indent_str(indent, ' ');
  auto msg = fmt::format("{}{}:0x{:016x}", indent_str,
                         toString(node->message.type), node->message.id);
  ImGui::Text("%s", msg.c_str());
  for (const auto &childNode : node->children) {
    displayNodeAndChildren(childNode);
  }
}

void createImGuiWindowsFromEventTree(const EventTree &eventTree) {
  for (const auto &node : eventTree.root->children) {
    if (node->message.type == MessageType::Process) {
      auto name = fmt::format("Process: {}", node->message.procid);
      ImGui::Begin(name.c_str());
      displayNodeAndChildren(node);
      ImGui::End();
    }
  }
}

void renderEventProcessWindows(RenderView &renderView) {
  auto &eventTree = GlobalState::getSingleton().getEventTree();
  createImGuiWindowsFromEventTree(eventTree);
}

void createFlameWindows(RenderView &renderView) {
  auto &state = globalState();

  static int generation = 0;
  if (generation == state.generation)
    return;
  generation = state.generation;

  auto &flameWindows = state.flameWindows;

  std::unordered_set<EventTreeNode::Ptr> roots;
  for (auto &node : state.getEventTree().root->children) {
    roots.insert(node);
  }

  for (auto &node : roots) {
    auto it = flameWindows.find(node);
    if (it == flameWindows.end()) {
      auto &wgpuDevice = renderView.renderWindow.renderCore.wgpuDevice;
      flameWindows[node] = std::make_unique<FlameWindow>(node, wgpuDevice);
    }
  }

  std::vector<EventTreeNode::Ptr> to_remove;
  for (auto &[node, flameWindow] : flameWindows) {
    if (roots.find(node) == roots.end()) {
      to_remove.push_back(node);
    }
  }

  for (auto &node : to_remove) {
    flameWindows.erase(node);
  }

  for (auto &[node, flameWindow] : flameWindows) {
    flameWindow->regenerate();
  }
}

void renderFlameWindows(RenderView &renderView) {
  auto &state = globalState();
  auto &flameWindows = state.flameWindows;

  for (auto &[node, flameWindow] : flameWindows) {
    TriangleWindow::TooltipCallback tooltipCallback;
    if (auto pickSpan = flameWindow->getPickSpan(); pickSpan) {
      tooltipCallback = [pickSpan]() {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
        ImGui::BeginTooltip();
        std::string name{pickSpan->getName()};
        ImGui::Text("%s", name.c_str());
        std::string idstr =
            fmt::format("id: 0x{:016x}", pickSpan->push->message.id);
        ImGui::Text("%s", idstr.c_str());
        auto dur = pickSpan->range.duration();
        auto durStr = dur.toString(Time::Precision::Milliseconds);
        ImGui::Text("dur: %s", durStr.c_str());
        if (auto locStr = pickSpan->locStr(); locStr) {
          ImGui::Text("loc: %s", locStr->data());
        }
        ImGui::EndTooltip();
        ImGui::PopStyleVar();
      };
    }

    flameWindow->triangleWindow->renderImgui(tooltipCallback);
  }
}

void preRenderPass(RenderView &renderView) {
  auto &state = globalState();
  state.frameStart = Time::Timestamp::now();

  if (!checkGeneration<1>())
    return;

  auto &eventTree = state.getEventTree();
  auto &eventTreeRoot = eventTree.root;
  auto &eventTreeRootChildren = eventTreeRoot->children;
  for (auto &child : eventTreeRootChildren) {
    if (child->message.type == MessageType::Process) {
      handleProcessNode(child);
    }
  }
}

void postRenderPass(RenderView &renderView) {
  if (!checkGeneration<2>())
    return;
}

void createDockSpace(RenderView &renderView) {
  static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

  ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

  ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                  ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
  window_flags |=
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("DockSpace", nullptr, window_flags);
  ImGui::PopStyleVar();
  ImGui::PopStyleVar(2);

  ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
  ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

  ImGui::End();
}

std::string getBasename(const EventTreeNode::Ptr &node) {
  const auto &msg = node->message;
  if (msg.id == 0) {
    if (node->children.empty()) {
      return "";
    }
    return getBasename(node->children[0]);
  }
  DateTime dt = DateTime::fromNanoseconds(msg.ts);
  std::string basename = fmt::format("motr.trace.{}", dt.toFilenameString());
  return basename;
}

template <typename T>
void logRPCReflectable(const T &result, std::string_view prefix) {
  result.reflect([&prefix](const auto &field, const auto &value) {
    MOTR_LOG("  {}.{}={}", prefix, field, value);
  });
}

void processEventTree(RenderView &renderView) {
  auto &state = globalState();
  auto &rpcResults = state.rpcResults;

  std::vector<EventTreeNode::Ptr> toKeep;
  std::vector<EventTreeNode::Ptr> remove;

  for (auto &rpcResult : rpcResults) {
    if (auto result = M::motr::RPC::getRPCResult<HostInfo>(rpcResult); result) {
      logRPCReflectable(*result, "HostInfo");
      remove.push_back(rpcResult);
    } else if (auto result =
                   M::motr::RPC::getRPCResult<MotrServerInfo>(rpcResult);
               result) {
      logRPCReflectable(*result, "MotrServerInfo");
      remove.push_back(rpcResult);
      auto now = M::motr::Time::Timestamp::now();
      auto startTimestamp = M::motr::Time::Timestamp(result->startTimestamp);
      auto buildTimestamp = M::motr::Time::Timestamp(result->buildTimestamp);
      auto elapsedSinceStart = now - startTimestamp;
      auto elapsedSinceBuild = now - buildTimestamp;
      MOTR_LOG("MotrServerInfo.startTimestamp={} ({} ago)",
               startTimestamp.toString(), elapsedSinceStart.toString());
      MOTR_LOG("MotrServerInfo.buildTimestamp={} ({} ago)",
               buildTimestamp.toString(), elapsedSinceBuild.toString());
    } else {
      MOTR_LOG("Unknown RPC: {}", M::motr::RPC::getFingerprint(rpcResult));
      toKeep.push_back(rpcResult);
    }
  }

  // Remove the RPC results that have been processed
  for (auto &rpcResult : remove) {
    rpcResult->setParent(nullptr);
  }

  state.rpcResults = toKeep;
}

} // namespace M::motr::Gui
