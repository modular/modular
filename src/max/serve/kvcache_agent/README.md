# KV Cache Agent Service

A gRPC service that provides real-time KV cache state updates.

## Overview

The KVCacheAgentService allows clients to subscribe to real-time updates about
KV cache entries as they are added or removed across different memory tiers
(GPU, CPU, disk, or object store).

## Usage

### Testing

```bash
# integration tests
bazelw test //SDK/integration-test/serve/kvcache_agent:tests --test_output=streamed --test_arg="-svv"
```
