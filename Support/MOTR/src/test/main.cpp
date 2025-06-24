//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#define MOTR_SINGLE_HEADER

#include "motr/Queue.h"
#include "motr/SharedMemory.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <stdio.h>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Simple C++17-compatible barrier for N threads
// TODO: use std::barrier when we can use C++20.
class Barrier {
public:
  explicit Barrier(int count) : threshold(count), count(count), generation(0) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mtx);
    auto gen = generation;
    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
    } else {
      cv.wait(lock, [this, gen] { return gen != generation; });
    }
  }

private:
  std::mutex mtx;
  std::condition_variable cv;
  int threshold;
  int count;
  int generation;
};

// This function tests the basic functionality of the queue.
// It checks if the queue is created successfully, if it is empty initially,
// sends a set of integers, verifies the sent count, checks if the queue is not
// empty, receives the messages, and ensures the received messages match the
// sent ones. Finally, it checks if the queue is empty after all messages are
// received. Create a queue for integers with a capacity of 10
void testQueue() {
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate, "/testQueue",
                            10);

  // Check if the queue is valid
  assert(queue.valid() && "Queue should be valid after creation");
  // Check if the queue is empty
  assert(queue.empty() && "Queue should be empty after creation");

  // Send some integers to the queue
  std::vector<int> messages = {1, 2, 3, 4, 5};
  const size_t sentCount = queue.send(messages.data(), messages.size());

  // Check if the correct number of messages were sent
  assert(sentCount == messages.size() && "All messages should be sent");

  // Check if the queue is no longer empty
  assert(!queue.empty() && "Queue should not be empty after sending messages");

  // Receive messages from the queue
  auto receivedMessages = queue.recv(5);

  // Check if the received messages match the sent messages
  assert(receivedMessages.size() == messages.size() &&
         "Received message count should match sent count");
  for (size_t i = 0; i < messages.size(); ++i) {
    assert(receivedMessages[i] == messages[i] &&
           "Received message should match sent message");
  }

  // Check if the queue is empty after receiving all messages
  assert(queue.empty() && "Queue should be empty after receiving all messages");
}

// This function tests the maximum capacity of the queue by sending items until
// the queue is full.
void stressTestMaximumCapacity() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/stressTestQueue", capacity);

  for (int i = 0; i < capacity; ++i) {
    assert(queue.send(&i, 1) == 1 && "Should send one item");
  }
  assert(queue.full() && "Queue should be full after sending maximum capacity");
}

// This function tests the queue's behavior when attempting to receive more
// items than sent.
void stressTestUnderflow() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/stressTestQueue", capacity);

  for (int i = 0; i < capacity; ++i) {
    assert(queue.send(&i, 1) == 1 && "Should send one item");
  }
  auto receivedMessages = queue.recv(capacity + 1);
  assert(receivedMessages.size() == capacity &&
         "Should receive maximum capacity items");
  assert(queue.empty() && "Queue should be empty after receiving all items");
}

// This function tests repeated send and receive operations to ensure the queue
// handles continuous operations correctly.
void stressTestRepeatedSendReceive() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/stressTestQueue", capacity);

  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < capacity; ++j) {
      assert(queue.send(&j, 1) == 1 && "Should send one item");
    }
    auto receivedMessages = queue.recv(capacity);
    assert(receivedMessages.size() == capacity &&
           "Should receive maximum capacity items");
    assert(queue.empty() && "Queue should be empty after receiving all items");
  }
}

// This function tests randomized sending and receiving of items and checks the
// behavior when receiving from an empty queue.
void stressTestRandomizedSendReceive() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/stressTestQueue", capacity);

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 5);
  for (int i = 0; i < 50; ++i) {
    int sendCount = distribution(generator);
    for (int j = 0; j < sendCount && !queue.full(); ++j) {
      assert(queue.send(&j, 1) == 1 && "Should send one item");
    }
    auto receivedMessages = queue.recv(sendCount);
    assert(receivedMessages.size() <= sendCount &&
           "Should receive up to sent count items");
  }
}

// This function tests the behavior of receiving from an empty queue.
void stressTestEmptyQueueReceive() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/stressTestQueue", capacity);

  for (int i = 0; i < 10; ++i) {
    auto receivedMessages = queue.recv(1);
    assert(receivedMessages.empty() &&
           "Receiving from an empty queue should return empty");
  }
}

// This function tests the queue's behavior when attempting to exceed its
// capacity. It fills the queue to its maximum capacity and then tries to send
// one more item. The test verifies that the send operation fails (returns 0)
// when the queue is full, ensuring that the queue does not accept more items
// than it can handle.
void testQueueSendWhileFull() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/overflowTestQueue", capacity);

  for (int i = 0; i < capacity; ++i) {
    assert(queue.send(&i, 1) == 1 && "Should send one item");
  }
  assert(queue.full() && "Queue should be full after sending maximum capacity");

  // Attempt to send one more item when queue is full
  int payload = 42;
  size_t sentCount = queue.send(&payload, 1);
  assert(sentCount == 0 && "Should not send item when queue is full");
}

// This function tests the queue's behavior in a multithreaded environment. It
// simulates multiple producers sending messages to the queue and a single
// consumer receiving messages from the queue concurrently.
void testQueueThreadSafety() {
  const size_t capacity = 32;      // Increased capacity
  const size_t numMessages = 1000; // More messages per producer
  const int NUM_PRODUCERS =
      std::max(4u, std::thread::hardware_concurrency()); // At least 4 producers
  const int NUM_CONSUMERS = 1;
  const size_t totalMessages = numMessages * NUM_PRODUCERS;

  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/threadSafetyTestQueue", capacity);

  std::atomic<int> sentCount(0);
  std::atomic<int> receivedCount(0);

  auto producer = [&]() {
    for (int i = 0; i < numMessages; ++i) {
      while (queue.send(&i, 1) != 1) {
        if (rand() % 4 == 0)
          std::this_thread::yield();
        if (rand() % 16 == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(rand() % 50));
      }
      sentCount++;
      if (rand() % 8 == 0)
        std::this_thread::yield();
    }
  };

  auto consumer = [&]() {
    while (true) {
      while (queue.recv(1).size() == 0) {
        if (rand() % 4 == 0)
          std::this_thread::yield();
        if (rand() % 16 == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(rand() % 50));
      }
      receivedCount++;
      if (receivedCount >= totalMessages)
        break;
      if (rand() % 8 == 0)
        std::this_thread::yield();
    }
  };

  std::vector<std::thread> producers(NUM_PRODUCERS);
  std::vector<std::thread> consumers(NUM_CONSUMERS);

  for (int i = 0; i < NUM_PRODUCERS; ++i)
    producers[i] = std::thread(producer);
  for (int i = 0; i < NUM_CONSUMERS; ++i)
    consumers[i] = std::thread(consumer);

  for (auto &producer : producers)
    producer.join();
  for (auto &consumer : consumers)
    consumer.join();

  // Verify that the number of sent and received messages match
  assert(sentCount == receivedCount && "Sent and received counts should match");
  assert(queue.empty() &&
         "Queue should be empty after all messages are processed");
}

void testQueueDifferentDataTypes() {
  // Test with strings
  {
    const size_t capacity = 10;
    M::motr::Queue<std::string> stringQueue(
        M::SharedMemoryInit::ExclusiveCreate, "/stringQueue", capacity);

    // Send strings
    std::vector<std::string> messages = {"Hello",     "World", "Test", "Queue",
                                         "Different", "Data",  "Types"};
    for (const auto &msg : messages) {
      size_t result = stringQueue.send(msg);
      assert(result == 1 && "Failed to send one string");
    }

    // Receive strings
    std::vector<std::string> receivedStrings =
        stringQueue.recv(messages.size());
    size_t receivedCount = receivedStrings.size();
    assert(receivedCount == messages.size() &&
           "Received a different number of strings than sent");
    for (size_t i = 0; i < receivedCount; ++i) {
      assert(receivedStrings[i] == messages[i] &&
             "Received string does not match sent string");
    }
  }
}

// This unit test verifies the behavior of the Queue API when the queue reaches
// its maximum capacity and then wraps around.  It begins by initializing a
// Queue instance for integers with a specified capacity of 5 using shared
// memory.  The test first fills the queue to its maximum capacity by sending 5
// integers. After that, it receives all items from the queue and checks that
// the number of received items matches the maximum capacity.  Next, the test
// attempts to send additional integers (from 5 to 9) to the queue to test the
// wrap-around functionality.  Finally, it receives items again and verifies
// that the number of received items matches the maximum capacity once more.
// This test is crucial for ensuring that the queue correctly handles
// wrap-around scenarios, allowing new items to be sent and received even after
// the queue has reached its capacity.
void testQueueWrapAround() {
  const size_t capacity = 5;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/wrapAroundQueue", capacity);

  // Fill the queue to its maximum capacity
  for (int i = 0; i < capacity; ++i) {
    assert(queue.send(&i, 1) == 1 && "Should send one item");
  }

  // Receive all items
  auto receivedItems = queue.recv(capacity);
  assert(receivedItems.size() == capacity &&
         "Should receive maximum capacity items");

  // Send more items to test wrap-around
  for (int i = capacity; i < capacity * 2; ++i) {
    assert(queue.send(&i, 1) == 1 && "Should send one item");
  }

  // Receive items again
  receivedItems = queue.recv(capacity);
  assert(receivedItems.size() == capacity &&
         "Should receive maximum capacity items");
}

// This unit test verifies the behavior of the Queue API when attempting to send
// a null pointer (nullptr) as a message.  It ensures that the queue correctly
// handles invalid input by rejecting the attempt to send a nullptr.
void testQueueSendNullptr() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate, "/nullQueue",
                            capacity);

  // Attempt to send a nullptr
  int *nullPtr = nullptr;
  size_t result = queue.send(nullPtr, 1);
  assert(result == 0 && "Should not send nullptr");
}

// This test verifies the behavior of the Queue API when attempting to receive
// more items than have been sent to the queue.  It begins by initializing a
// Queue instance for integers with a specified capacity of 10 using shared
// memory.  Next, it sends a known number of integers (5 in this case) to the
// queue. After that, the test attempts to receive a larger number of items (10)
// than what has been sent to the queue.  Finally, it captures the received
// items and checks that the number of items returned matches the number of
// items that were actually sent (5).  This test is important for validating
// that the queue implementation does not allow over-receiving and correctly
// limits the number of items returned to the consumer based on what has been
// sent, ensuring the integrity and reliability of the queue's behavior in
// scenarios where consumers may request more data than is available.
void testQueueReceiveWithMaxCount() {
  const size_t capacity = 10;
  M::motr::Queue<int> queue(M::SharedMemoryInit::ExclusiveCreate,
                            "/maxCountQueue", capacity);

  // Send some integers
  for (int i = 0; i < 5; ++i) {
    assert(queue.send(&i, 1) == 1 && "Should send one item");
  }

  // Attempt to receive more items than sent
  auto receivedItems = queue.recv(10); // Requesting more than sent
  assert(receivedItems.size() == 5 &&
         "Should receive only the number of items sent");
}

// Test Sending std::string_view
// void testQueueSendStringView() {
//     const size_t capacity = 10;
//     M::motr::Queue<std::string> queue(M::SharedMemoryInit::ExclusiveCreate,
//     "/stringViewQueue", capacity);
//
//     // Prepare a vector of string views
//     std::vector<std::string_view> messages = {
//         "Hello", "World", "Test", "Queue", "StringView", "Overload",
//         "Example"
//     };
//
//     // Send string views to the queue
//     int sent = queue.send(messages);
//     assert(sent == 1 && "Should send one string view");
//
//     // Receive strings from the queue
//     std::vector<std::string> receivedStrings = queue.recv(messages.size());
//     assert(receivedStrings.size() == messages.size() && "Should receive the
//     same number of strings");
//
//     // Verify that the received strings match the sent string views
//     for (size_t i = 0; i < receivedStrings.size(); ++i) {
//         assert(receivedStrings[i] == messages[i] && "Received string does not
//         match sent string view");
//     }
// }

void testQueueABAWriteheadRaceDeterministic() {
  using namespace M::motr;
  using namespace M;

  constexpr size_t capacity = 2;
  Queue<int> queue(SharedMemoryInit::ExclusiveCreate, "test_queue_aba",
                   capacity);

  Barrier ready_barrier(3); // 2 producers + main
  std::atomic<bool> p1_success{false};
  std::atomic<bool> p2_success{false};

  auto producer = [&](int value, std::atomic<bool> &success_flag) {
    // Both producers check available space before proceeding
    size_t available = queue.numAvailableToWrite();
    assert(available == capacity); // Both should see 2
    ready_barrier.wait();          // Wait for both to reach this point

    // Both try to send 2 elements (should only be possible for one)
    std::vector<int> data = {value, value + 1};
    size_t sent = queue.send(data.data(), data.size());
    success_flag = (sent == data.size());
  };

  std::thread t1(producer, 100, std::ref(p1_success));
  std::thread t2(producer, 200, std::ref(p2_success));

  // Main thread waits for both producers to be ready
  ready_barrier.wait();

  t1.join();
  t2.join();

  // Only one producer should have succeeded, not both
  int num_success = int(p1_success) + int(p2_success);
  if (num_success > 1) {
    std::cerr << "ABA bug detected: both producers succeeded, buffer overrun "
                 "possible!\n";
    assert(false && "ABA bug: both producers over-committed the buffer");
  }

  // Consume and check data integrity
  auto result = queue.recv(4);

  // Check: exactly 2 elements
  assert(result.size() == 2 &&
         "Queue should contain exactly 2 elements after ABA test");

  // Check: only one producer's data is present
  bool valid1 = (result[0] == 100 && result[1] == 101);
  bool valid2 = (result[0] == 200 && result[1] == 201);
  assert((valid1 || valid2) &&
         "Queue contains invalid or mixed data after ABA test");

  // Check: queue is empty after read
  assert(queue.empty() && "Queue should be empty after ABA test");
}

void testSharedMemoryCleanup() {
  using namespace M;
  const std::string shm_name = "/test_shm_cleanup";
  constexpr size_t shm_size = 4096;

  // Step 1: Create and write a value
  {
    SharedMemory shm(SharedMemoryInit::ExclusiveCreate, shm_name, shm_size);
    assert(shm.valid());
    std::memset(shm.data, 0xAB, shm_size); // Fill with known value
  } // shm goes out of scope, destructor called, should cleanup

  // Step 2: Re-create with the same name, should not see old value
  {
    SharedMemory shm(SharedMemoryInit::ExclusiveCreate, shm_name, shm_size);
    assert(shm.valid());
    // Check that the memory is not filled with 0xAB
    unsigned char *bytes = static_cast<unsigned char *>(shm.data);
    bool all_ab = true;
    for (size_t i = 0; i < shm_size; ++i) {
      if (bytes[i] != 0xAB) {
        all_ab = false;
        break;
      }
    }
    assert(!all_ab && "Shared memory was not cleaned up properly!");
    std::cout << "Shared memory cleanup test passed.\n";
  }
}

void testSharedMemoryAutoCleanupOnCrash() {
  using namespace M;
  const std::string shm_name = "/test_shm_auto_cleanup";
  constexpr size_t shm_size = 4096;

  pid_t pid = fork();
  if (pid == 0) {
    // Child process: create and fill the segment, then crash
    SharedMemory shm(SharedMemoryInit::ExclusiveCreate, shm_name, shm_size);
    assert(shm.valid());
    std::memset(shm.data, 0xCD, shm_size); // Fill with known value
    std::cout << "[child] Created and filled shared memory, now crashing...\n";
    std::abort(); // Simulate crash (SIGABRT)
  } else {
    // Parent process: wait for child to exit
    int status = 0;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && "Child did not terminate by signal");

    // Now try to create the same segment again
    SharedMemory shm(SharedMemoryInit::ExclusiveCreate, shm_name, shm_size);
    assert(shm.valid());
    // Check that the memory is not filled with 0xCD
    unsigned char *bytes = static_cast<unsigned char *>(shm.data);
    bool all_cd = true;
    for (size_t i = 0; i < shm_size; ++i) {
      if (bytes[i] != 0xCD) {
        all_cd = false;
        break;
      }
    }
    assert(!all_cd && "Shared memory was not cleaned up after crash!");
    std::cout << "Shared memory auto-cleanup on crash test passed.\n";
  }
}

void testMultiProcessProducerConsumer() {
  using namespace M;
  using namespace M::motr;
  const std::string shm_name = "/test_mp_queue";
  constexpr size_t capacity = 4096;
  const int num_producers = std::max(2u, std::thread::hardware_concurrency());
  constexpr int messages_per_producer = 1000;
  const int total_messages = num_producers * messages_per_producer;

  printf("  [info] Using %d producer processes (hardware concurrency)\n",
         num_producers);

  // Parent: create the queue
  Queue<int> queue(SharedMemoryInit::ExclusiveCreate, shm_name, capacity);

  // Fork producers
  std::vector<pid_t> pids;
  for (int p = 0; p < num_producers; ++p) {
    pid_t pid = fork();
    if (pid == 0) {
      // Child: producer
      Queue<int> q(SharedMemoryInit::OpenExisting, shm_name, capacity);
      int base = p * messages_per_producer;
      for (int i = 0; i < messages_per_producer; ++i) {
        int value = base + i;
        while (q.send(&value, 1) != 1) {
          std::this_thread::sleep_for(
              std::chrono::microseconds(10)); // Wait for space
        }
      }
      _exit(0);
    }
    pids.push_back(pid);
  }

  // Fork consumer
  pid_t consumer_pid = fork();
  if (consumer_pid == 0) {
    Queue<int> q(SharedMemoryInit::OpenExisting, shm_name, capacity);
    std::set<int> received;
    while (received.size() < total_messages) {
      auto vals = q.recv(64);
      for (int v : vals) {
        if (!received.insert(v).second) {
          std::cerr << "Duplicate value: " << v << std::endl;
          _exit(2);
        }
      }
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    // Check all expected values are present
    for (int p = 0; p < num_producers; ++p) {
      int base = p * messages_per_producer;
      for (int i = 0; i < messages_per_producer; ++i) {
        int v = base + i;
        if (received.count(v) == 0) {
          std::cerr << "Missing value: " << v << std::endl;
          _exit(3);
        }
      }
    }
    std::cout << "All values received correctly.\n";
    _exit(0);
  }

  // Parent: wait for all children
  int status = 0;
  for (pid_t pid : pids)
    waitpid(pid, &status, 0);
  waitpid(consumer_pid, &status, 0);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
         "Multi-process producer/consumer test failed!");
  std::cout << "Multi-process producer/consumer test passed.\n";
}

int main() {

  // printf("Testing shared memory auto-cleanup on crash...\n");
  // testSharedMemoryAutoCleanupOnCrash();
  // printf("✓ Shared memory auto-cleanup on crash test passed\n");

  printf("Testing shared memory cleanup...\n");
  testSharedMemoryCleanup();
  printf("✓ Shared memory cleanup test passed\n");

  constexpr int max_iterations = 10;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    printf("Running basic queue tests...\n");
    testQueue();
    printf("✓ Basic queue test passed\n");

    printf("Testing queue behavior when full...\n");
    testQueueSendWhileFull();
    printf("✓ Full queue test passed\n");

    printf("Testing different data types...\n");
    testQueueDifferentDataTypes();
    printf("✓ Data type tests passed\n");

    printf("Testing queue wrap around...\n");
    testQueueWrapAround();
    printf("✓ Wrap around test passed\n");

    printf("Testing nullptr handling...\n");
    testQueueSendNullptr();
    printf("✓ Nullptr test passed\n");

    printf("Testing receive with max count...\n");
    testQueueReceiveWithMaxCount();
    printf("✓ Max count test passed\n");

    // testQueueSendStringView();

    printf("Running thread safety tests...\n");
    constexpr int maxThreadSafetyIteration = 25;
    for (int i = 0; i < maxThreadSafetyIteration; i++) {
      testQueueThreadSafety();
      if (i % 10 == 0)
        printf("  Thread safety test %d/%d\n", i, maxThreadSafetyIteration);
    }
    printf("✓ Thread safety tests passed\n");

    printf("Running stress tests...\n");
    printf("  Testing maximum capacity...\n");
    stressTestMaximumCapacity();
    printf("  ✓ Maximum capacity test passed\n");

    printf("  Testing underflow...\n");
    stressTestUnderflow();
    printf("  ✓ Underflow test passed\n");

    printf("  Testing repeated send/receive...\n");
    stressTestRepeatedSendReceive();
    printf("  ✓ Repeated send/receive test passed\n");

    printf("  Testing randomized send/receive...\n");
    stressTestRandomizedSendReceive();
    printf("  ✓ Randomized send/receive test passed\n");

    printf("  Testing empty queue receive...\n");
    stressTestEmptyQueueReceive();
    printf("  ✓ Empty queue receive test passed\n");

    printf("  Testing ABA writehead race...\n");
    constexpr int maxABAStressIterations = 100;
    for (int i = 0; i < 100; i++) {
      testQueueABAWriteheadRaceDeterministic();
      if (i % 10 == 0)
        printf("  ✓ ABA writehead race test passed %d/%d\n", i,
               maxABAStressIterations);
    }
    printf("  ✓ ABA writehead race test passed\n");

    printf("Testing multi-process producer/consumer...\n");
    testMultiProcessProducerConsumer();
    printf("✓ Multi-process producer/consumer test passed\n");

    printf("%d%%\n", (iteration + 1) * 100 / max_iterations);
  }

  printf("All tests passed!\n");
  return 0;
}
