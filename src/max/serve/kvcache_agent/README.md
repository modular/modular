# KV Cache Agent Service

A gRPC service that provides real-time KV cache state updates.

## Overview

The KVCacheAgentService allows clients to subscribe to real-time updates about
KV cache entries as they are added or removed across different memory tiers
(GPU, CPU, disk, or object store).

## Usage

### Starting the Service using Bazel

```bash
bazelw run \
    //SDK/lib/API/python/max/serve/kvcache_agent:kvcache_agent_server -- \
    --host=0.0.0.0 --port=50051 --workers=4
```
