# Memory Manager Unit Test Plan

## Overview

This document outlines a comprehensive testing plan for the AsyncRT
MemoryManager system. The memory manager is a sophisticated memory management
component that requires thorough testing to ensure correctness, performance, and
reliability.

## Current State

The memory manager implementation consists of three main components:

- **MemoryManager**: Main cache managing chunks and descriptors
- **MemoryManagerFreeList**: Hierarchical free list for allocation/deallocation
- **Supporting classes**: Chunk, Desc, DescMap for internal management

Test coverage now spans roughly 80 tests across eight files under
`MLRT/unittests/Driver/` (`MemoryManagerTest.cpp`,
`MemoryManagerChunkGrowthTest.cpp`, the `MemoryManagerVMM*` suites, and the
`GPUMemoryManager*` suites), in addition to the original allocation-pattern
coverage in `DeviceContextTest.cpp`. This document tracks the remaining gaps
against the components and behaviors enumerated below.

## Key Components to Test

### Core Classes

- `MemoryManager`: Memory pool manager with configurable chunk and cache sizes
- `MemoryManagerFreeList`: Hierarchical free list with parent-child
  relationships
- `MemoryManager::Chunk`: Individual memory chunks allocated from device
- `MemoryManager::Desc`: Descriptors tracking memory regions
- `MemoryManager::DescMap`: Hash map for fast pointer-to-descriptor lookup

### Critical Functionality

- Chunk allocation and management
- Descriptor lifecycle and recycling
- Free list hierarchy and delegation
- Memory alignment handling
- Block splitting and merging
- Configuration-driven behavior

## Proposed Test Structure

### 1. Basic Functionality Tests

#### `MemoryManagerCreation`

- Test factory methods (`onDevice`, `onHost`) with various configurations
- Verify proper initialization of cache parameters
- Test configuration parsing from environment variables
- Edge cases: zero sizes, invalid configurations

#### `ChunkAllocation`

- Test initial chunk allocation
- Verify chunk growth heuristics
- Test maximum cache size limits
- Memory pressure scenarios
- Device memory availability constraints

#### `DescriptorManagement`

- Test descriptor allocation and recycling
- Verify descriptor pool management (`popDesc`/`pushDesc`)
- Test descriptor reset and reuse
- Descriptor map operations (add/remove)

#### `AlignmentHandling`

- Test various alignment requirements (64, 128, 256, etc.)
- Verify alignment mask calculations
- Edge cases with unusual alignment values
- Interaction between alignment and chunk boundaries

### 2. Free List Management Tests

#### `FreeListHierarchy`

- Test parent-child relationships
- Verify delegation to parent free lists
- Test free list creation and destruction
- Multiple hierarchy levels

#### `FreeListFlushing`

- Test flushing to parent on destruction
- Verify proper block merging during flush
- Test flush with various block sizes and patterns
- Error handling during flush operations

#### `FreeListSorting`

- Verify free blocks maintained in size order
- Test insertion at various positions
- Verify removal preserves ordering
- Edge cases: single block, identical sizes

#### `FreeListMerging`

- Test merging with previous block only
- Test merging with next block only
- Test merging with both previous and next blocks
- Test no-merge scenarios
- Verify proper descriptor cleanup after merging

### 3. Memory Management Tests

#### `AllocationPatterns`

- Sequential allocations of increasing sizes
- Random allocation patterns
- Fragmentation-inducing patterns
- Large allocation scenarios
- Small allocation scenarios

#### `DeallocationSequences`

- LIFO (stack-like) deallocation
- FIFO (queue-like) deallocation
- Random deallocation order
- Interleaved allocation/deallocation
- Verify coalescing in all scenarios

#### `SplitAndMerge`

- Block splitting for exact size matches
- Block splitting with remainder
- Merging of adjacent free blocks
- Complex scenarios with multiple splits and merges
- Verify descriptor accounting

#### `MemoryReuse`

- Verify freed blocks are properly reused
- Test best-fit allocation strategy
- Test allocation from split blocks
- Verify no memory leaks in reuse cycles

### 4. Configuration & Heuristics Tests

#### `ChunkSizeGrowth`

- Test `growChunkSizeTo` functionality
- Verify heuristics for chunk size growth
- Test `enableUseMaxSize` behavior
- Test `shouldUseMaxSize` static flag
- Configuration-driven growth patterns

#### `MaxSizeHandling`

- Test maximum cache size enforcement
- Verify behavior when approaching limits
- Test allocation failure at max size
- Memory pressure response

#### `ConfigurationVariations`

- Different cache sizes (small, medium, large)
- Various chunk size percentages (1%, 25%, 50%, 100%)
- Host vs device configurations
- Invalid configuration handling

#### `MemoryPressure`

- Behavior under low memory conditions
- Graceful degradation patterns
- Fallback to direct device allocation
- Recovery after memory pressure relief

### 5. Error Handling & Edge Cases

#### `AllocationFailures`

- Device allocation failures
- Out-of-memory scenarios
- Fallback behavior verification
- Error propagation testing

#### `InvalidOperations`

- Double free detection/handling
- Invalid pointer handling
- Null pointer operations
- Operations on destroyed cache

#### `BoundaryConditions`

- Zero-size allocation requests
- Maximum size allocation requests
- Allocation exactly at chunk boundaries
- Allocation larger than chunk size

#### `ConfigurationLimits`

- Invalid cache sizes (negative, zero)
- Invalid chunk percentages (>100%, negative)
- Invalid alignment values (non-power-of-2)
- Environment variable edge cases

### 6. Thread Safety & Concurrency

#### `ConcurrentAllocations`

- Multiple threads allocating simultaneously
- Verify proper mutex protection
- Test allocation performance under contention
- Deadlock prevention

#### `ConcurrentFrees`

- Multiple threads freeing simultaneously
- Verify thread-safe free list operations
- Test descriptor map thread safety
- Concurrent flush operations

#### `MixedOperations`

- Concurrent allocation and free operations
- Cross-thread allocation and free
- Hierarchical free list contention
- Performance under mixed workloads

#### `LockContention`

- Verify minimal lock scope
- Test lock acquisition ordering
- Measure contention under stress
- Validate atomic operations where used

### 7. Debugging & Validation Tests

#### `SelfCheckValidation`

- Enable self-check via environment variable
- Verify all invariant checks trigger properly
- Test self-check performance impact
- Hierarchical self-check verification

#### `LoggingOutput`

- Enable logging via environment variable
- Verify expected log message format
- Test logging performance impact
- Log message content validation

#### `MemoryLeakDetection`

- Verify all chunks are released on destruction
- Test descriptor pool cleanup
- Verify no dangling pointers
- Integration with memory debugging tools

#### `InvariantViolations`

- Deliberately corrupt data structures (in debug builds)
- Verify assertion triggers
- Test recovery mechanisms
- Validate error messages

### 8. Performance & Metrics Tests

#### `AllocationSpeed`

- Measure allocation time vs direct device allocation
- Compare performance across different cache configurations
- Benchmark allocation patterns
- Memory access pattern optimization

#### `FragmentationMeasurement`

- Track fragmentation under various allocation patterns
- Measure fragmentation over time
- Compare fragmentation with different strategies
- External fragmentation analysis

#### `CacheEfficiency`

- Measure cache hit rates
- Track memory utilization percentages
- Analyze allocation success rates
- Compare efficiency metrics across configurations

#### `ScalingBehavior`

- Performance with increasing cache sizes
- Scalability with number of concurrent threads
- Memory usage growth patterns
- Degradation analysis under stress

### 9. Integration Tests

#### `DeviceContextIntegration`

- Verify proper integration with DeviceContext
- Test buffer creation through DeviceContext
- Verify allocation mode handling
- Cross-component error propagation

#### `BufferLifecycle`

- End-to-end buffer creation, use, and destruction
- Integration with DeviceBuffer classes
- Memory transfer operations with cached buffers
- Resource cleanup verification

#### `MemoryTransfers`

- Verify cached buffers work with copy operations
- Test host-to-device transfers
- Test device-to-device transfers
- Performance comparison with non-cached buffers

#### `MultipleContexts`

- Multiple DeviceContexts with separate caches
- Shared cache scenarios
- Resource isolation verification
- Cross-context interference testing

## Test Implementation Strategy

### Test File Organization

- Create `MemoryManagerTest.cpp` as dedicated test file
- Use Google Test framework for consistency
- Organize tests into logical test fixtures
- Implement helper classes for common operations

### Test Utilities

- Mock Device classes for controlled testing
- Memory pattern generators for predictable data
- Performance measurement utilities
- Configuration management helpers

### Environment Setup

- Scoped environment variable management
- Temporary file and resource cleanup
- Thread synchronization utilities
- Memory debugging integration

### Validation Helpers

- Invariant checking utilities
- Memory pattern verification
- Performance threshold validation
- Resource leak detection

## Success Criteria

### Coverage Goals

- 100% line coverage of MemoryManager implementation
- 95%+ branch coverage including error paths
- All public API methods tested
- All configuration combinations validated

### Quality Metrics

- Zero memory leaks detected
- All thread safety scenarios validated
- Performance regressions detected
- Comprehensive error path testing

### Maintenance

- Tests should be maintainable and readable
- Clear documentation of test intent
- Minimal test interdependencies
- Easy addition of new test scenarios

## Risk Mitigation

### Potential Issues

- Complex memory management bugs
- Race conditions in concurrent scenarios
- Configuration interaction edge cases
- Performance regressions

### Mitigation Strategies

- Comprehensive invariant checking
- Stress testing under various loads
- Systematic configuration matrix testing
- Performance baseline establishment

## Future Considerations

### Extensibility

- Framework for adding new allocation strategies
- Support for different memory types
- Integration with additional debugging tools
- Performance monitoring enhancements

### Maintenance

- Regular review of test effectiveness
- Updates for new features or configurations
- Performance baseline updates
- Documentation maintenance

---

This comprehensive test plan will significantly improve the reliability and
maintainability of the AsyncRT memory manager system by providing thorough,
focused testing of its critical functionality.
