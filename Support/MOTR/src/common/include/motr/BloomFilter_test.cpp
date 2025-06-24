//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BloomFilter.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

// Forward declaration of fnv1a from BloomFilter.cpp
uint64_t fnv1a(const uint8_t *data, size_t len);

// Forward declarations
void testFilterFalsePositives(const BloomFilter &bf,
                              const std::vector<uint64_t> &trainData,
                              const std::vector<uint64_t> &testData,
                              double expectedFpr,
                              const std::string &filterType);

// Helper function to hash strings using fnv1a
uint64_t hashString(const std::string &str) {
  return fnv1a(reinterpret_cast<const uint8_t *>(str.c_str()), str.length());
}

// Generate random 64-bit values spanning the full range
std::vector<uint64_t> generateRandomValues(size_t count, uint64_t seed = 42) {
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<uint64_t> dis;

  std::vector<uint64_t> values;
  values.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    values.push_back(dis(gen));
  }

  return values;
}

// Test basic functionality with small dataset
void testBasicFunctionality() {
  std::cout << "=== Testing Basic Functionality ===" << std::endl;

  BloomFilter bf(100, 0.01);

  // Verify filter parameters are reasonable
  assert(bf.m > 0 && "Bloom filter must have bits allocated!");
  assert(bf.k > 0 && "Bloom filter must have hash functions!");
  assert(bf.bits.size() > 0 && "Bloom filter bits array must not be empty!");

  std::vector<uint64_t> data = {1, 2, 3, 4, 5};

  // Test that empty filter doesn't have items
  for (auto item : data) {
    assert(!bf.has(item) && "Empty bloom filter should not contain any items!");
  }

  bf.add(data);

  // Test for false negatives (should never happen)
  for (auto item : data) {
    assert(bf.has(item) && "False negative detected!");
  }

  std::vector<uint64_t> test = {3, 4, 5, 6, 7};
  auto inside = bf.intersect(test);
  auto outside = bf.difference(test);

  std::cout << "Intersect: ";
  for (auto x : inside)
    std::cout << x << " ";
  std::cout << "\nDifference: ";
  for (auto x : outside)
    std::cout << x << " ";
  std::cout << std::endl;

  // Validate that we don't have 100% false positive rate
  assert(outside.size() > 0 &&
         "Bloom filter appears broken - no items in difference set!");

  std::cout << "Basic functionality test PASSED" << std::endl << std::endl;
}

// Test serialization/deserialization
void testSerialization() {
  std::cout << "=== Testing Serialization ===" << std::endl;

  BloomFilter bf(100, 0.01);
  std::vector<uint64_t> data = {1, 2, 3, 4, 5};
  bf.add(data);

  auto raw = bf.serialize();
  BloomFilter bf2 = BloomFilter::deserialize(raw);

  // Test for false negatives after deserialization
  for (auto x : data) {
    assert(bf2.has(x) && "False negative after deserialization!");
  }

  std::cout << "Serialization test PASSED" << std::endl << std::endl;
}

// Test with different sizes and measure false positive rates with strict
// validation
void testFalsePositiveRates(size_t dataSize, size_t testSize,
                            double expectedFpr = 0.01) {
  std::cout << "=== Testing " << dataSize
            << " elements (FPR target: " << expectedFpr << ") ===" << std::endl;

  BloomFilter bf(dataSize, expectedFpr);

  // Calculate and print memory usage
  size_t memoryUsage =
      sizeof(bf) + bf.bits.size() + sizeof(bf.m) + sizeof(bf.k);
  std::cout << "Memory usage: " << memoryUsage << " bytes ("
            << (memoryUsage / 1024.0) << " KB)" << std::endl;
  std::cout << "Bits array size: " << bf.bits.size() << " bytes" << std::endl;
  std::cout << "Parameters: m=" << bf.m << ", k=" << bf.k << std::endl;

  // Generate data spanning 64-bit space
  auto trainData = generateRandomValues(dataSize, 42);
  auto testData = generateRandomValues(testSize, 123);

  // Add training data to bloom filter
  bf.add(trainData);

  // Test original filter
  std::cout << "\n--- Testing Original Filter ---" << std::endl;
  testFilterFalsePositives(bf, trainData, testData, expectedFpr, "Original");

  // Serialize and deserialize
  auto serializedData = bf.serialize();
  BloomFilter deserializedBf = BloomFilter::deserialize(serializedData);

  std::cout << "Serialized data size: " << serializedData.size() << " bytes"
            << std::endl;

  // Verify deserialized filter has same parameters
  assert(deserializedBf.m == bf.m && "Deserialized m parameter mismatch!");
  assert(deserializedBf.k == bf.k && "Deserialized k parameter mismatch!");
  assert(deserializedBf.bits.size() == bf.bits.size() &&
         "Deserialized bits size mismatch!");

  // Test deserialized filter
  std::cout << "\n--- Testing Deserialized Filter ---" << std::endl;
  testFilterFalsePositives(deserializedBf, trainData, testData, expectedFpr,
                           "Deserialized");

  std::cout << "Test with " << dataSize << " elements PASSED" << std::endl
            << std::endl;
}

// Helper function to test false positives on a specific filter
void testFilterFalsePositives(const BloomFilter &bf,
                              const std::vector<uint64_t> &trainData,
                              const std::vector<uint64_t> &testData,
                              double expectedFpr,
                              const std::string &filterType) {

  // Verify no false negatives
  for (auto item : trainData) {
    assert(bf.has(item) &&
           ("False negative detected in " + filterType + " filter!").c_str());
  }

  // Test for false positives
  std::set<uint64_t> trainSet(trainData.begin(), trainData.end());
  size_t falsePositives = 0;
  std::vector<uint64_t> fpItems;

  for (auto item : testData) {
    if (bf.has(item) && trainSet.find(item) == trainSet.end()) {
      falsePositives++;
      fpItems.push_back(item);
    }
  }

  double actualFpr = static_cast<double>(falsePositives) / testData.size();

  std::cout << filterType << " filter results:" << std::endl;
  std::cout << "Data size: " << trainData.size() << std::endl;
  std::cout << "Test size: " << testData.size() << std::endl;
  std::cout << "False positives: " << falsePositives << std::endl;
  std::cout << "Actual FPR: " << actualFpr << std::endl;
  std::cout << "Expected FPR: " << expectedFpr << std::endl;

  if (!fpItems.empty()) {
    std::cout << "False positive items (first 10): ";
    for (size_t i = 0; i < std::min(fpItems.size(), size_t(10)); ++i) {
      std::cout << fpItems[i] << " ";
    }
    std::cout << std::endl;
  }

  // Validate FPR is within reasonable bounds (allow 5x tolerance for small
  // datasets)
  double tolerance =
      std::max(5.0, 1.0 / std::sqrt(static_cast<double>(testData.size())));
  double maxAllowedFpr = expectedFpr * tolerance;
  double minAllowedFpr = expectedFpr / tolerance;

  assert(
      actualFpr <= maxAllowedFpr &&
      ("False positive rate too high in " + filterType + " filter!").c_str());
  assert(actualFpr >= minAllowedFpr &&
         ("False positive rate suspiciously low in " + filterType + " filter!")
             .c_str());
  assert(actualFpr < 0.5 &&
         ("False positive rate indicates broken " + filterType + " filter!")
             .c_str());

  std::cout << "FPR validation: " << minAllowedFpr << " <= " << actualFpr
            << " <= " << maxAllowedFpr << std::endl;
}

// Test m and k parameter calculations
void testParameterCalculations() {
  std::cout << "=== Testing Parameter Calculations ===" << std::endl;

  struct TestCase {
    uint64_t capacity;
    double fpr;
    uint64_t expectedMinM;
    uint64_t expectedMaxM;
    uint64_t expectedMinK;
    uint64_t expectedMaxK;
  };

  std::vector<TestCase> testCases = {// capacity, fpr, minM, maxM, minK, maxK
                                     {1, 0.01, 8, 20, 1, 10},
                                     {10, 0.01, 90, 100, 6, 8},
                                     {100, 0.01, 900, 1000, 6, 8},
                                     {1000, 0.01, 9000, 10000, 6, 8},
                                     {100, 0.001, 1400, 1500, 9, 11},
                                     {100, 0.1, 400, 500, 3, 5},
                                     {1000000, 0.01, 9000000, 10000000, 6, 8}};

  for (const auto &tc : testCases) {
    BloomFilter bf(tc.capacity, tc.fpr);

    std::cout << "Capacity: " << tc.capacity << ", FPR: " << tc.fpr
              << " -> m: " << bf.m << ", k: " << bf.k << std::endl;

    // Validate m is in reasonable range
    assert(bf.m >= tc.expectedMinM && "m parameter too small!");
    assert(bf.m <= tc.expectedMaxM && "m parameter too large!");

    // Validate k is in reasonable range
    assert(bf.k >= tc.expectedMinK && "k parameter too small!");
    assert(bf.k <= tc.expectedMaxK && "k parameter too large!");

    // Validate bits array size matches m
    uint64_t expectedBitsSize = (bf.m + 7) / 8;
    assert(bf.bits.size() == expectedBitsSize &&
           "Bits array size doesn't match m!");

    // Validate parameters are positive
    assert(bf.m > 0 && "m must be positive!");
    assert(bf.k > 0 && "k must be positive!");
  }

  std::cout << "Parameter calculations test PASSED" << std::endl << std::endl;
}

// Test wide variety of BloomFilter instantiations
void testVariousInstantiations() {
  std::cout << "=== Testing Various Instantiations ===" << std::endl;

  std::vector<uint64_t> capacities = {1,   2,    5,     10,     50,     100,
                                      500, 1000, 10000, 100000, 1000000};
  std::vector<double> fprs = {0.001, 0.01, 0.05, 0.1, 0.2, 0.3};

  size_t testCount = 0;
  for (auto capacity : capacities) {
    for (auto fpr : fprs) {
      BloomFilter bf(capacity, fpr);

      // Basic sanity checks
      assert(bf.m > 0 && "m must be positive!");
      assert(bf.k > 0 && "k must be positive!");
      assert(bf.bits.size() > 0 && "bits array must not be empty!");
      assert(bf.bits.size() == (bf.m + 7) / 8 && "bits array size mismatch!");

      // Check that m scales roughly with capacity
      if (capacity >= 100) {
        assert(bf.m >= capacity &&
               "m should be at least capacity for reasonable FPR!");
      }

      // Check that k is reasonable (typically 1-20 for practical filters)
      assert(bf.k <= 50 && "k seems unreasonably large!");

      // Verify theoretical FPR calculation
      double theoreticalFpr = std::pow(
          1.0 - std::exp(-static_cast<double>(bf.k * capacity) / bf.m), bf.k);
      assert(theoreticalFpr <= fpr * 2.0 &&
             "Theoretical FPR much higher than requested!");

      testCount++;
    }
  }

  std::cout << "Tested " << testCount << " different instantiations"
            << std::endl;
  std::cout << "Various instantiations test PASSED" << std::endl << std::endl;
}

// Test with string hashing
void testStringHashing() {
  std::cout << "=== Testing String Hashing ===" << std::endl;

  // Generate 50 different strings
  std::vector<std::string> strings;
  for (int i = 0; i < 50; ++i) {
    strings.push_back("test_string_" + std::to_string(i) + "_unique_suffix_" +
                      std::to_string(i * 7));
  }

  // Hash all strings
  std::vector<uint64_t> hashedStrings;
  for (const auto &str : strings) {
    hashedStrings.push_back(hashString(str));
  }

  // Put first half into bloom filter
  std::vector<uint64_t> trainHashes(hashedStrings.begin(),
                                    hashedStrings.begin() + 25);
  std::vector<uint64_t> testHashes(hashedStrings.begin() + 25,
                                   hashedStrings.end());

  BloomFilter bf(25, 0.01);
  bf.add(trainHashes);

  // Verify no false negatives for training data
  for (auto hash : trainHashes) {
    assert(bf.has(hash) && "False negative for string hash!");
  }

  // Test the second half
  size_t falsePositives = 0;
  std::vector<std::string> fpStrings;

  for (size_t i = 0; i < testHashes.size(); ++i) {
    if (bf.has(testHashes[i])) {
      falsePositives++;
      fpStrings.push_back(strings[25 + i]);
    }
  }

  double actualFpr = static_cast<double>(falsePositives) / testHashes.size();

  std::cout << "String test results:" << std::endl;
  std::cout << "Training strings: 25" << std::endl;
  std::cout << "Test strings: 25" << std::endl;
  std::cout << "False positives: " << falsePositives << std::endl;
  std::cout << "Actual FPR: " << actualFpr << std::endl;

  // Validate FPR is reasonable (allow higher tolerance for small sample)
  assert(actualFpr <= 0.5 &&
         "String test FPR too high - filter may be broken!");

  if (!fpStrings.empty()) {
    std::cout << "False positive strings: ";
    for (const auto &str : fpStrings) {
      std::cout << "\"" << str << "\" ";
    }
    std::cout << std::endl;
  }

  // Test intersect and difference with string hashes
  auto inside = bf.intersect(testHashes);
  auto outside = bf.difference(testHashes);

  std::cout << "Intersect count: " << inside.size() << std::endl;
  std::cout << "Difference count: " << outside.size() << std::endl;

  assert(inside.size() + outside.size() == testHashes.size() &&
         "Intersect + difference should equal total!");

  std::cout << "String hashing test PASSED" << std::endl << std::endl;
}

// Test edge cases
void testEdgeCases() {
  std::cout << "=== Testing Edge Cases ===" << std::endl;

  // Test with single element
  BloomFilter bf1(1, 0.01);
  std::vector<uint64_t> singleItem = {0xFFFFFFFFFFFFFFFFULL}; // Max uint64_t
  bf1.add(singleItem);
  assert(bf1.has(0xFFFFFFFFFFFFFFFFULL) && "Failed with max uint64_t!");

  // Test with zero
  BloomFilter bf2(1, 0.01);
  std::vector<uint64_t> zeroItem = {0};
  bf2.add(zeroItem);
  assert(bf2.has(0) && "Failed with zero!");

  // Test with powers of 2
  BloomFilter bf3(10, 0.01);
  std::vector<uint64_t> powers;
  for (int i = 0; i < 64; ++i) {
    powers.push_back(1ULL << i);
  }
  bf3.add(powers);

  for (auto power : powers) {
    assert(bf3.has(power) && "Failed with power of 2!");
  }

  // Test edge case parameters
  BloomFilter bf4(0, 0.01); // Should handle capacity = 0
  assert(bf4.m > 0 && bf4.k > 0 && "Should handle zero capacity!");

  BloomFilter bf5(100, 0.0); // Should handle invalid FPR
  assert(bf5.m > 0 && bf5.k > 0 && "Should handle zero FPR!");

  BloomFilter bf6(100, 1.0); // Should handle invalid FPR
  assert(bf6.m > 0 && bf6.k > 0 && "Should handle FPR = 1.0!");

  std::cout << "Edge cases test PASSED" << std::endl << std::endl;
}

int main() {
  std::cout << "Running comprehensive BloomFilter tests..." << std::endl
            << std::endl;

  testBasicFunctionality();
  testSerialization();
  testParameterCalculations();
  testVariousInstantiations();

  // Test different sizes with varying false positive rates
  testFalsePositiveRates(10, 1000, 0.01);
  testFalsePositiveRates(100, 10000, 0.01);
  testFalsePositiveRates(1000, 100000, 0.01);
  testFalsePositiveRates(1000, 100000, 0.001);   // Lower FPR
  testFalsePositiveRates(1000, 100000, 0.00001); // Lower FPR

  testStringHashing();
  testEdgeCases();

  std::cout << "=== ALL TESTS PASSED ===" << std::endl;
  return 0;
}
