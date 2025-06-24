//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/BloomFilter.h"
#include <cstring>
#include <fstream>
#include <iostream>

uint64_t fnv1a(const uint8_t *data, size_t len) {
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

BloomFilter::BloomFilter(uint64_t capacity, double false_positive_rate) {
  if (capacity == 0)
    capacity = 1;
  if (false_positive_rate <= 0.0 || false_positive_rate >= 1.0)
    false_positive_rate = 0.01;

  double ln2 = std::log(2.0);

  // Calculate optimal number of bits: m = -n * ln(p) / (ln(2)^2)
  double mDouble = -static_cast<double>(capacity) *
                   std::log(false_positive_rate) / (ln2 * ln2);
  m = static_cast<uint64_t>(std::ceil(mDouble));

  // Calculate optimal number of hash functions: k = (m/n) * ln(2)
  double kDouble =
      (static_cast<double>(m) / static_cast<double>(capacity)) * ln2;
  k = static_cast<uint64_t>(std::round(kDouble));

  // Ensure minimum values
  if (m == 0)
    m = 8; // At least 1 byte
  if (k == 0)
    k = 1; // At least 1 hash function

  bits.resize((m + 7) / 8, 0);
}

void BloomFilter::add(const std::vector<uint64_t> &items) {
  for (uint64_t item : items) {
    uint64_t h1 = fnv1a(reinterpret_cast<const uint8_t *>(&item), sizeof(item));
    for (uint64_t i = 0; i < k; ++i) {
      uint64_t h = h1 ^ (i * 0x9e3779b97f4a7c15ULL);
      uint64_t bit = h % m;
      bits[bit / 8] |= (1 << (bit % 8));
    }
  }
}

bool BloomFilter::has(uint64_t item) const {
  uint64_t h1 = fnv1a(reinterpret_cast<const uint8_t *>(&item), sizeof(item));
  for (uint64_t i = 0; i < k; ++i) {
    uint64_t h = h1 ^ (i * 0x9e3779b97f4a7c15ULL);
    uint64_t bit = h % m;
    if ((bits[bit / 8] & (1 << (bit % 8))) == 0)
      return false;
  }
  return true;
}

std::vector<uint8_t> BloomFilter::serialize() const {
  std::vector<uint8_t> data;
  data.reserve(sizeof(m) + sizeof(k) + sizeof(uint64_t) + bits.size());

  auto append = [&](const void *ptr, size_t size) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(ptr);
    data.insert(data.end(), p, p + size);
  };

  uint64_t size = bits.size();
  append(&m, sizeof(m));
  append(&k, sizeof(k));
  append(&size, sizeof(size));
  append(bits.data(), bits.size());

  return data;
}

BloomFilter BloomFilter::deserialize(const std::vector<uint8_t> &rawdata) {
  BloomFilter bf(1);
  size_t offset = 0;

  auto read = [&](void *dst, size_t size) {
    std::memcpy(dst, rawdata.data() + offset, size);
    offset += size;
  };

  uint64_t size;
  read(&bf.m, sizeof(bf.m));
  read(&bf.k, sizeof(bf.k));
  read(&size, sizeof(size));
  bf.bits.resize(size);
  read(bf.bits.data(), size);
  return bf;
}

std::vector<uint64_t>
BloomFilter::intersect(const std::vector<uint64_t> &candidates) const {
  std::vector<uint64_t> results;
  for (auto x : candidates) {
    if (has(x))
      results.push_back(x);
  }
  return results;
}

std::vector<uint64_t>
BloomFilter::difference(const std::vector<uint64_t> &candidates) const {
  std::vector<uint64_t> results;
  for (auto x : candidates) {
    if (!has(x))
      results.push_back(x);
  }
  return results;
}
