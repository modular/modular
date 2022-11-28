//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/BroadcastShape.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace M;

ErrorOr<TensorShape> M::broadcastedShape(const TensorShape &a,
                                         const TensorShape &b) {
  // On some platforms ssize_t and int64_t are defined differently.
  BroadcastShape bcast(SmallVector<int64_t>(a.begin(), a.end()),
                       SmallVector<int64_t>(b.begin(), b.end()),
                       /*fewerDimsOptimization=*/false);

  if (bcast.isValid())
    return TensorShape(bcast.getResultShape());
  return Error("Incompatible shapes between " + a.getAsString() + " and " +
               b.getAsString());
}

void M::computeBatchIndices(int64_t outputBatchSize, ArrayRef<int64_t> reshape,
                            ArrayRef<int64_t> bcast,
                            llvm::SmallVectorImpl<int64_t> &out_indices) {
  // Populate the mapping in out_indices. This algorithm is identical to the
  // following steps:
  //  - Reshape {0, 1, ..., input_batch_size - 1} to the input shape.
  //  - Broadcast to the output shape.
  //  - Reshape back to a flat 1D vector.
  out_indices.resize(outputBatchSize);
  int64_t num_output_elements = 1;
  int64_t num_input_elements = 1;
  for (int64_t i = reshape.size() - 1; i >= 0; --i) {
    // Replicate the already populated mapping an additional (dim - 1) times. If
    // we are broadcasting, just copy the existing mapping. Otherwise, add
    // another dimension from the input shape.
    const int64_t dim = std::max(reshape[i], bcast[i]);
    const int64_t incr = bcast[i] > 1 ? 0 : num_input_elements;
    for (int64_t k = 0; k < (dim - 1) * num_output_elements; ++k)
      out_indices[num_output_elements + k] = out_indices[k] + incr;
    num_output_elements *= dim;
    num_input_elements *= reshape[i];
  }
}

template <int N>
M::BCastList<N>::BCastList(ArrayRef<ArrayRef<int64_t>> x,
                           bool fewerDimsOptimization) {
  using Vec = BCastList::Vec;
  for (int i = 0; i < N; ++i) {
    reshape.emplace_back(Vec());
    bcast.emplace_back(Vec());
    batch_indices_.emplace_back(SmallVector<int64_t>());
  }

  // Safely multiplies dimensions taking into account symbolic shapes.
  auto mul_dims = [](int64_t dim1, int64_t dim2) -> int64_t {
    return dim1 != 0 && dim2 != 0 && (dim1 < 0 || dim2 < 0) ? -1 : dim1 * dim2;
  };

  bool all_equal = true;
  size_t largest_rank = 0;
  outputBatchSize = 1;
  for (int i = 0; i < N; ++i) {
    all_equal = all_equal && x[i] == x[0];
    largest_rank = std::max(largest_rank, x[i].size());
  }
  if (all_equal)
    broadcastingRequired = false;
  if (all_equal && fewerDimsOptimization) {
    // Fast path for common case of identical shapes.
    int64_t elements = 1;
    const int rank = x[0].size();
    output.resize(rank);
    for (int i = 0; i < rank; i++) {
      const int64_t dim = x[0][i];
      elements = mul_dims(elements, dim);
      output[i] = dim;
    }
    result.push_back(elements);
    outputBatchSize = elements;
    for (int i = 0; i < N; ++i) {
      reshape[i].push_back(elements);
      bcast[i].push_back(1);
    }
    return;
  }

  // Reverse all the shapes for convenience
  // After the reverse, 0-th is the inner-most dimension.
  SmallVector<Vec, N> copy;
  for (int i = 0; i < N; ++i) {
    copy.emplace_back(x[i]);
    reverse(copy[i]);
  }

  // 1-extend and align all vectors.
  for (int i = 0; i < N; ++i)
    if (copy[i].size() < largest_rank)
      copy[i].resize(largest_rank, 1);

  // Going through each dimension starting from the inner-most
  // dimension, compares dimension of x and y. They are compatible if
  // they are equal or either is 1.

  // indices of j-th component of each input.
  bool prev_is_one[N];
  bool current_is_one[N];
  for (int i = 0; i < N; ++i) {
    prev_is_one[i] = false;
    current_is_one[i] = false;
  }
  bool output_dim_set = false;
  int output_dim = -1;
  bool set_one = false;
  for (size_t j = 0; j < largest_rank; ++j) {
    output_dim = -1;
    output_dim_set = false;
    // Find which indices are 1.
    for (int i = 0; i < N; ++i) {
      // Keep track of which indices are 1.
      if (copy[i][j] == 1) {
        current_is_one[i] = true;
      } else {
        current_is_one[i] = false;
        if (!output_dim_set || copy[i][j] == output_dim) {
          output_dim = copy[i][j];
          output_dim_set = true;
        } else {
          valid = false;
          return;
        }
      }
    }
    output.push_back(output_dim_set ? output_dim : 1);
    outputBatchSize = mul_dims(outputBatchSize, output.back());
    // All dimensions are 1.
    if (!output_dim_set) {
      if (!fewerDimsOptimization) {
        for (int i = 0; i < N; ++i) {
          bcast[i].push_back(1);
          reshape[i].push_back(1);
        }
        result.push_back(1);
      }

      // This will skip updating the previous state to the current one. We'll
      // explain why this is safe below.
      // Consider the previous state P, current state C and the next state N.
      // In the case where N also is all ones (N == C), we'll do the same
      // optimization here (push back one dimensions if we need to), which is
      // safe and is expected.
      //
      // When N != C, we'll continue as usual. However, we might trigger the
      // next block if N == P (because we didn't update the previous state).
      // We trigger the next block if `fewerDimsOptimization` is true.
      // This means that we did not modify and broadcast / reshapes in this
      // block (we skipped updating, since the one dimensions can be ignored).
      // In essence, we only need to check whether the previous non-one state is
      // equal to the current non-one state.

      continue;
    } else if (fewerDimsOptimization &&
               std::equal(current_is_one, current_is_one + N, prev_is_one) &&
               set_one) {
      // It is a run of the same broadcasting case as last time.
      // We can reshape the input so that fewer dimensions
      // are involved in the intermediate computation.
      result.back() = mul_dims(result.back(), output_dim);
      for (int i = 0; i < N; ++i) {
        reshape[i].back() = mul_dims(reshape[i].back(), copy[i][j]);
        bcast[i].back() =
            mul_dims(bcast[i].back(), current_is_one[i] ? output_dim : 1);
      }
    } else {
      result.push_back(output_dim);
      for (int i = 0; i < N; ++i) {
        reshape[i].push_back(copy[i][j]);
        bcast[i].push_back(current_is_one[i] ? output_dim : 1);
      }
    }
    set_one = true;
    for (int i = 0; i < N; ++i)
      prev_is_one[i] = current_is_one[i];
  }
  if (result.empty()) {
    result.push_back(1);
    for (int i = 0; i < N; ++i) {
      reshape[i].push_back(1);
      bcast[i].push_back(1);
    }
  }
  // Do something about batches.
  for (int i = 0; i < N; ++i) {
    reverse(reshape[i]);
    reverse(bcast[i]);
  }
  reverse(result);
  reverse(output);
}

M::BroadcastShape::BroadcastShape(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                                  bool fewerDimsOptimization)
    : BCastList<2>(SmallVector<ArrayRef<int64_t>, 2>({x, y}),
                   fewerDimsOptimization) {}
