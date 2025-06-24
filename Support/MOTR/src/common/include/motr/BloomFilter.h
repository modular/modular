//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_BLOOM_FILTER_H
#define MOTR_BLOOM_FILTER_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct BloomFilter {
  std::vector<uint8_t> bits;
  uint64_t m; // number of bits
  uint64_t k; // number of hash functions

  BloomFilter(uint64_t capacity, double false_positive_rate = 0.01);

  void add(const std::vector<uint64_t> &items);
  bool has(uint64_t item) const;
  std::vector<uint8_t> serialize() const;
  static BloomFilter deserialize(const std::vector<uint8_t> &rawdata);

  std::vector<uint64_t>
  intersect(const std::vector<uint64_t> &candidates) const;
  std::vector<uint64_t>
  difference(const std::vector<uint64_t> &candidates) const;
};
#endif // MOTR_BLOOM_FILTER_H
