//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#define MOTR_SINGLE_HEADER

#include "motr/Queue.h"
#include "motr/SharedMemory.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

using namespace M;
using namespace M::motr;

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

struct BenchConfig {
  const size_t queueCapacity;         // Size of queue in messages
  const size_t numMessages;           // Number of messages per test
  const size_t maxProducers;          // Max number of producers for MPSC
  const size_t numWarmupIters;        // Warmup iterations before measurement
  const size_t numTrials;             // Number of measurement trials
  const std::vector<size_t> msgSizes; // Message sizes for bandwidth test

  static BenchConfig getDefault() {
    return {
        /* queueCapacity */ 1024 * 64,
        /* numMessages */ 1000000,
        /* maxProducers */
        static_cast<size_t>(std::max(4u, std::thread::hardware_concurrency())),
        /* numWarmupIters */ 3,
        /* numTrials */ 5,
        /* msgSizes */ {8, 64, 256, 1024, 4096, 16384, 65536}};
  }
};

//------------------------------------------------------------------------------
// Statistics Collection
//------------------------------------------------------------------------------

struct BenchStats {
  double min, max, avg;

  explicit BenchStats(const std::vector<double> &samples) {
    min = *std::min_element(samples.begin(), samples.end());
    max = *std::max_element(samples.begin(), samples.end());
    avg = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  }

  void print(const char *metric, const char *unit) const {
    printf("%s: %.2f%s [%.2f-%.2f]\n", metric, avg, unit, min, max);
  }
};

struct BatchStats {
  size_t single_ops = 0; // Operations that processed exactly 1 message
  size_t batch_ops = 0;  // Operations that processed >1 message
  size_t batch_msgs = 0; // Total messages processed in batch operations

  double avg_batch_size() const {
    if (batch_ops == 0)
      return 0.0;
    return static_cast<double>(batch_msgs) / static_cast<double>(batch_ops);
  }

  double batch_ratio() const {
    size_t total_ops = single_ops + batch_ops;
    if (total_ops == 0)
      return 0.0;
    return static_cast<double>(batch_ops) / static_cast<double>(total_ops);
  }

  void print(const char *) const {
    printf("batch: %.1f msgs/op (%.0f%% batched)\n", avg_batch_size(),
           batch_ratio() * 100.0);
  }
};

struct BackoffStats {
  size_t immediate_success = 0;
  size_t yield_count = 0;
  size_t total_ops = 0;

  double success_ratio() const {
    return total_ops ? double(immediate_success) / total_ops : 0.0;
  }

  double avg_yields() const {
    return total_ops ? double(yield_count) / total_ops : 0.0;
  }

  void print(const char *) const {
    printf("contention: %.0f%% success, %.1f yields/op\n",
           success_ratio() * 100.0, avg_yields());
  }
};

//------------------------------------------------------------------------------
// Benchmark Implementation
//------------------------------------------------------------------------------

class QueueBenchmark {
  const BenchConfig &config;
  struct TestResult {
    double throughput;
    double batch_size;
    double batch_ratio;
  };
  std::vector<std::pair<std::string, TestResult>> results;

public:
  explicit QueueBenchmark(const BenchConfig &cfg) : config(cfg) {}

  void run() {
    printf("queue bench (cap:%zu msgs, iters:%zu+%zu warmup)\n",
           config.queueCapacity, config.numTrials, config.numWarmupIters);

    bench_spsc_throughput();

    // Test with increasing number of producers
    for (size_t n = config.maxProducers / 8; n <= config.maxProducers; n *= 2) {
      bench_mpsc_throughput(n);
    }

    bench_latency_contention();
    print_summary();

    bench_bandwidth();
  }

private:
  void store_result(const char *name, double throughput,
                    const BatchStats &batch_stats) {
    results.push_back({name,
                       {throughput, batch_stats.avg_batch_size(),
                        batch_stats.batch_ratio()}});
  }

  void print_summary() {
    print_separator();
    printf("\nComparative Summary:\n\n");

    // Throughput summary
    printf("Throughput Scaling:\n");
    double baseline = results[0].second.throughput;
    for (const auto &result : results) {
      printf("%-7s %.2fM msgs/s (%.0f%% of baseline)\n", result.first.c_str(),
             result.second.throughput,
             (result.second.throughput / baseline) * 100);
    }

    printf("\nBatch Efficiency:\n");
    for (const auto &result : results) {
      printf("%-7s %.1f msgs/op (%.0f%% batched)\n", result.first.c_str(),
             result.second.batch_size, result.second.batch_ratio * 100);
    }
  }

  void print_separator() const {
    printf("\n-----------------------------------------------\n");
  }

  void bench_bandwidth() {
    print_separator();
    printf("bandwidth test:\n");

    for (size_t msg_size : config.msgSizes) {
      printf("\n%zuB msgs: ", msg_size);

      alignas(64) char send_buf[1048576];
      memset(send_buf, 0x42, sizeof(send_buf));
      std::vector<double> bandwidths;

      for (size_t trial = 0; trial < config.numWarmupIters + config.numTrials;
           trial++) {
        Queue<char> q(SharedMemoryInit::ExclusiveCreate, "bench_bw",
                      config.queueCapacity);
        // assert(q.valid());

        std::atomic<bool> done{false};
        size_t total_bytes = config.numMessages * msg_size;

        std::thread consumer([&]() {
          size_t received = 0;
          std::vector<char> recv_buf(msg_size);
          while (received < total_bytes) {
            auto msgs = q.recv(msg_size);
            received += msgs.size();
          }
          done = true;
        });

        // Pre-warm
        q.send(send_buf, msg_size);
        q.recv(msg_size);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config.numMessages;) {
          size_t sent = q.send(send_buf, msg_size);
          if (sent)
            ++i;
        }
        while (!done)
          std::this_thread::yield();
        auto end = std::chrono::high_resolution_clock::now();
        consumer.join();

        double sec = std::chrono::duration<double>(end - start).count();
        double gbps = (total_bytes / 1e9) / sec;

        if (trial >= config.numWarmupIters) {
          bandwidths.push_back(gbps);
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      BenchStats stats(bandwidths);
      stats.print("GB/s", "");
    }
  }

  void bench_spsc_throughput() {
    print_separator();
    printf("spsc test:\n");

    std::vector<double> throughputs;
    BatchStats batch_stats;
    BackoffStats backoff_stats;

    // Standard test
    run_spsc_variant(throughputs, batch_stats, backoff_stats, "standard",
                     false);
    store_result("SPSC:", throughputs.back(), batch_stats);

    // Reset stats for wait strategy variant
    throughputs.clear();
    batch_stats = BatchStats{};
    backoff_stats = BackoffStats{};

    // Wait strategy test
    run_spsc_variant(throughputs, batch_stats, backoff_stats, "wait strategy",
                     true);
    store_result("SPSC-W:", throughputs.back(), batch_stats);
  }

  void run_spsc_variant(std::vector<double> &throughputs,
                        BatchStats &batch_stats, BackoffStats &backoff_stats,
                        const char *variant, bool use_wait_strategy) {
    for (size_t trial = 0; trial < config.numWarmupIters + config.numTrials;
         trial++) {
      Queue<int> q(SharedMemoryInit::ExclusiveCreate, "bench_spsc",
                   config.queueCapacity);
      assert(q.valid());

      std::atomic<bool> done{false};
      std::atomic<size_t> pending_msgs{0}; // Track pending messages

      std::thread consumer([&]() {
        size_t received = 0;
        constexpr size_t BATCH_TARGET = 8;   // Target batch size
        constexpr size_t MAX_WAIT_NS = 1000; // Max 1μs wait

        while (received < config.numMessages) {
          if (use_wait_strategy) {
            // Wait for potential batch accumulation
            size_t wait_time = 0;
            while (pending_msgs.load(std::memory_order_relaxed) <
                       BATCH_TARGET &&
                   wait_time < MAX_WAIT_NS) {
              std::this_thread::sleep_for(std::chrono::nanoseconds(50));
              wait_time += 50;
            }
          }

          auto msgs = q.recv(config.numMessages);
          size_t batch_size = msgs.size();
          received += batch_size;

          if (batch_size > 0) {
            pending_msgs.fetch_sub(batch_size, std::memory_order_relaxed);
          }

          if (trial >= config.numWarmupIters) {
            if (batch_size == 1) {
              batch_stats.single_ops++;
            } else if (batch_size > 1) {
              batch_stats.batch_ops++;
              batch_stats.batch_msgs += batch_size;
            }
          }
        }
        done = true;
      });

      // Pre-warm
      int msg = 0;
      q.send(&msg, 1);
      q.recv(1);
      std::atomic_thread_fence(std::memory_order_seq_cst);

      auto start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < config.numMessages;) {
        int msg = static_cast<int>(i);
        size_t sent = q.send(&msg, 1);
        if (sent) {
          ++i;
          pending_msgs.fetch_add(1, std::memory_order_relaxed);
          if (trial >= config.numWarmupIters) {
            backoff_stats.immediate_success++;
          }
        } else if (trial >= config.numWarmupIters) {
          backoff_stats.yield_count++;
        }
        if (trial >= config.numWarmupIters)
          backoff_stats.total_ops++;
      }
      while (!done)
        std::this_thread::yield();
      auto end = std::chrono::high_resolution_clock::now();
      consumer.join();

      double sec = std::chrono::duration<double>(end - start).count();
      double throughput = config.numMessages / 1e6 / sec;

      if (trial >= config.numWarmupIters) {
        throughputs.push_back(throughput);
      }

      std::atomic_thread_fence(std::memory_order_seq_cst);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    BenchStats stats(throughputs);
    printf("\n%s:\n", variant);
    stats.print("throughput", "M msgs/s");
    batch_stats.print("");
    backoff_stats.print("");
  }

  void bench_mpsc_throughput(size_t numProducers) {
    print_separator();
    printf("mpsc test (%zu producers):\n", numProducers);

    std::vector<double> throughputs;
    BatchStats batch_stats;
    BackoffStats backoff_stats;
    std::vector<BackoffStats> producer_stats(numProducers);

    for (size_t trial = 0; trial < config.numWarmupIters + config.numTrials;
         trial++) {
      Queue<int> q(SharedMemoryInit::ExclusiveCreate, "bench_mpsc",
                   config.queueCapacity);
      assert(q.valid());

      std::atomic<size_t> produced{0};
      std::atomic<bool> done{false};
      std::vector<std::thread> producers;

      // Pre-warm
      int msg = 0;
      q.send(&msg, 1);
      q.recv(1);
      std::atomic_thread_fence(std::memory_order_seq_cst);

      for (size_t p = 0; p < numProducers; ++p) {
        producers.emplace_back([&, p]() {
          BackoffStats &stats = producer_stats[p];
          while (true) {
            size_t idx = produced.fetch_add(1);
            if (idx >= config.numMessages)
              break;
            int msg = static_cast<int>(idx);
            bool success = q.send(&msg, 1);
            if (trial >= config.numWarmupIters) {
              if (success)
                stats.immediate_success++;
              else
                stats.yield_count++;
              stats.total_ops++;
            }
            while (!success) {
              std::this_thread::yield();
              success = q.send(&msg, 1);
              if (trial >= config.numWarmupIters) {
                stats.yield_count++;
                stats.total_ops++;
              }
            }
          }
        });
      }

      std::thread consumer([&]() {
        size_t received = 0;
        while (received < config.numMessages) {
          auto msgs = q.recv(config.numMessages);
          received += msgs.size();
          if (trial >= config.numWarmupIters) {
            if (msgs.size() == 1) {
              batch_stats.single_ops++;
            } else if (msgs.size() > 1) {
              batch_stats.batch_ops++;
              batch_stats.batch_msgs += msgs.size();
            }
          }
        }
        done = true;
      });

      auto start = std::chrono::high_resolution_clock::now();
      for (auto &t : producers)
        t.join();
      while (!done)
        std::this_thread::yield();
      auto end = std::chrono::high_resolution_clock::now();
      consumer.join();

      double sec = std::chrono::duration<double>(end - start).count();
      double throughput = config.numMessages / 1e6 / sec;

      if (trial >= config.numWarmupIters) {
        throughputs.push_back(throughput);
      }

      std::atomic_thread_fence(std::memory_order_seq_cst);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    BenchStats stats(throughputs);
    stats.print("throughput", "M msgs/s");
    batch_stats.print("");

    char name[32];
    snprintf(name, sizeof(name), "MPSC%zu:", numProducers);
    store_result(name, stats.avg, batch_stats);
  }

  void bench_latency_contention() {
    print_separator();
    printf("latency test:\n");

    constexpr size_t N = 100000;
    std::vector<std::vector<uint64_t>> trial_latencies;
    BackoffStats backoff_stats;

    for (size_t trial = 0; trial < config.numWarmupIters + config.numTrials;
         trial++) {
      Queue<int> q(SharedMemoryInit::ExclusiveCreate, "bench_lat",
                   config.queueCapacity);
      assert(q.valid());

      std::vector<uint64_t> latencies;
      latencies.reserve(N);
      std::atomic<bool> done{false};

      // Pre-warm
      int msg = 0;
      q.send(&msg, 1);
      q.recv(1);
      std::atomic_thread_fence(std::memory_order_seq_cst);

      std::thread consumer([&]() {
        size_t received = 0;
        while (received < N) {
          auto msgs = q.recv(1);
          if (!msgs.empty())
            ++received;
        }
        done = true;
      });

      for (size_t i = 0; i < N; ++i) {
        int msg = static_cast<int>(i);
        auto t0 = std::chrono::high_resolution_clock::now();
        bool success = q.send(&msg, 1);
        if (trial >= config.numWarmupIters) {
          if (success)
            backoff_stats.immediate_success++;
          else
            backoff_stats.yield_count++;
          backoff_stats.total_ops++;
        }
        while (!success) {
          std::this_thread::yield();
          success = q.send(&msg, 1);
          if (trial >= config.numWarmupIters) {
            backoff_stats.yield_count++;
            backoff_stats.total_ops++;
          }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count());
      }

      while (!done)
        std::this_thread::yield();
      consumer.join();

      if (trial >= config.numWarmupIters) {
        trial_latencies.push_back(std::move(latencies));
      }

      std::atomic_thread_fence(std::memory_order_seq_cst);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::vector<double> p50s, p99s, maxes;
    for (const auto &latencies : trial_latencies) {
      auto sorted = latencies;
      std::sort(sorted.begin(), sorted.end());
      p50s.push_back(sorted[size_t(N * 0.5)]);
      p99s.push_back(sorted[size_t(N * 0.99)]);
      maxes.push_back(sorted.back());
    }

    BenchStats p50_stats(p50s);
    BenchStats p99_stats(p99s);
    BenchStats max_stats(maxes);

    p50_stats.print("p50", "ns");
    p99_stats.print("p99", "ns");
    max_stats.print("max", "ns");
    backoff_stats.print("");
  }
};

//------------------------------------------------------------------------------
// Main Entry Point
//------------------------------------------------------------------------------

int main() {
  printf("size_t is %zu bytes\n", sizeof(size_t));

  auto config = BenchConfig::getDefault();
  QueueBenchmark bench(config);
  bench.run();
  return 0;
}
