//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_REUSEGRAPHBLOBBUFFERS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// The first and last member of a graph blob that use a value, counting members
// in the order the blob runs them.
struct MemberRange {
  unsigned first;
  unsigned last;
};

std::optional<MemberRange> unionRanges(std::optional<MemberRange> a,
                                       std::optional<MemberRange> b) {
  if (!a || !b)
    return a ? a : b;
  return MemberRange{std::min(a->first, b->first), std::max(a->last, b->last)};
}

// Finds the members that each value in a graph blob reaches. Every op in a
// blob's body builds operands for its calls, so a value's members are those of
// the calls it feeds.
class BlobMembers {
public:
  explicit BlobMembers(GraphBlobOp blob) {
    for (auto [index, call] :
         llvm::enumerate(blob.getBody().getOps<CallExecuteDeviceOp>()))
      members[call] = MemberRange{static_cast<unsigned>(index),
                                  static_cast<unsigned>(index)};
  }

  std::optional<MemberRange> of(Value value) {
    std::optional<MemberRange> range;
    for (Operation *user : value.getUsers())
      range = unionRanges(range, of(user));
    return range;
  }

private:
  std::optional<MemberRange> of(Operation *op) {
    if (auto it = members.find(op); it != members.end())
      return it->second;
    std::optional<MemberRange> range;
    for (Value result : op->getResults())
      range = unionRanges(range, of(result));
    members[op] = range;
    return range;
  }

  DenseMap<Operation *, std::optional<MemberRange>> members;
};

// A buffer some of a blob routine's outputs are written to.
struct Buffer {
  // The output that owns it: the first to use it.
  unsigned owner;
  Type type;
  // The last member that uses it.
  unsigned lastUse;
  // Whether a later output may take it: false once an output that something
  // outside the routine reads is written to it.
  bool reusable;
};

void reuseBuffers(RoutineOp routine, GraphBlobOp blob) {
  RoutineOp::BodyArgs args = routine.getBodyArgs();
  ResultRange results = routine.getOutputChain()
                            ? routine.getOutputs().drop_back()
                            : routine.getOutputs();

  BlobMembers members(blob);
  SmallVector<MemberRange> ranges;
  for (BlockArgument pointer : args.outputs) {
    std::optional<MemberRange> range = members.of(pointer);
    assert(range && "a member writes every output of a graph blob");
    ranges.push_back(*range);
  }

  // Linear scan: visit the outputs in the order their first members run, and
  // give each the first free buffer of its type.
  SmallVector<unsigned> order =
      llvm::to_vector(llvm::seq<unsigned>(0, results.size()));
  llvm::stable_sort(order, [&](unsigned a, unsigned b) {
    return ranges[a].first < ranges[b].first;
  });
  SmallVector<Buffer> buffers;
  SmallVector<unsigned> bufferOf(results.size());
  for (unsigned output : order) {
    Type type = results[output].getType();
    bool readOutside = !results[output].use_empty();
    // Outputs of one type have one size, so each fits the other's buffer.
    auto free = llvm::find_if(buffers, [&](const Buffer &buffer) {
      return buffer.reusable && buffer.type == type &&
             buffer.lastUse < ranges[output].first;
    });
    if (free == buffers.end()) {
      bufferOf[output] = buffers.size();
      buffers.push_back(
          Buffer{output, type, ranges[output].last, !readOutside});
      continue;
    }
    bufferOf[output] = free - buffers.begin();
    free->lastUse = ranges[output].last;
    free->reusable = !readOutside;
  }
  if (buffers.size() == results.size())
    return;

  // Members write each output to its buffer owner's pointer, and the routine
  // returns one result per buffer.
  Block &body = routine.getBody().front();
  llvm::BitVector dropped(body.getNumArguments());
  for (auto [output, buffer] : llvm::enumerate(bufferOf)) {
    unsigned owner = buffers[buffer].owner;
    if (owner == output)
      continue;
    args.outputs[output].replaceAllUsesWith(args.outputs[owner]);
    dropped.set(args.outputs[output].getArgNumber());
  }

  SmallVector<Type> resultTypes = llvm::map_to_vector(
      buffers, [](const Buffer &buffer) { return buffer.type; });
  if (Value chain = routine.getOutputChain())
    resultTypes.push_back(chain.getType());
  OpBuilder builder(routine);
  auto reused = RoutineOp::create(
      builder, routine.getLoc(), resultTypes, routine.getInputs(),
      routine.getChain(), routine.getFusionInfoAttr(),
      routine.getOutputParamDeclsAttr(), routine.getHasReadEffectAttr(),
      routine.getHasWriteEffectAttr(), routine.getDispatchAttr());
  reused->setDiscardableAttrs(routine->getDiscardableAttrDictionary());
  reused.getBody().takeBody(routine.getBody());
  reused.getBody().front().eraseArguments(dropped);

  for (auto [output, buffer] : llvm::enumerate(bufferOf))
    results[output].replaceAllUsesWith(reused.getOutputs()[buffer]);
  if (Value chain = routine.getOutputChain())
    chain.replaceAllUsesWith(reused.getOutputChain());
  routine.erase();
}

struct ReuseGraphBlobBuffersPass
    : public impl::ReuseGraphBlobBuffersBase<ReuseGraphBlobBuffersPass> {
  void runOnOperation() override {
    SmallVector<std::pair<RoutineOp, GraphBlobOp>> blobs;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto routine = dyn_cast<RoutineOp>(op);
      if (!routine)
        return WalkResult::advance();
      if (auto blob = dyn_cast<GraphBlobOp>(routine.getBody().front().front()))
        blobs.emplace_back(routine, blob);
      return WalkResult::skip();
    });
    for (auto [routine, blob] : blobs)
      reuseBuffers(routine, blob);
  }
};

} // namespace

} // namespace M::MOGG
