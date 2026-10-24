//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//! Pinned-host <-> device copy engine (full-duplex).
//!
//! The Rust equivalent of `BlockOffloadEngine`. MAX allocates the pinned host
//! buffer (`Buffer` with `Usage.STAGING`) and passes its base address plus each leaf's
//! row width (`bytes_per_leaf`); this engine does the per-block H2D/D2H copies
//! against it and fans an MLA-replicated block out to its peer devices with
//! p2p copies (instead of the NVLink broadcast). It drives the GPU through the
//! vendor-agnostic [`crate::gpu`] shim, so it works on both NVIDIA (CUDA) and
//! AMD (HIP).
//!
//! Duplex: each device has TWO aux streams — an onload (H2D) stream and an
//! offload (D2H) stream — so the onload and offload lanes each own their own
//! stream. They never enqueue on a shared stream (no lock needed), and because
//! the copy engines / PCIe are full-duplex the H2D and D2H overlap on the GPU.
//!
//! A leaf's block `bid` lives at region byte offset
//! `bid * bytes_per_leaf[leaf_idx]`, which [`HostRegion`] resolves to a host
//! address in whichever chunk holds it; within that row each device
//! unit occupies `[offset, offset + stride)` and maps to device address
//! `unit.ptr + bid * stride`, where `stride` is that unit's own `bytes`.

use std::collections::HashMap;

use itertools::Itertools as _;

use crate::{
    error::{ConnectorError, Result},
    gpu::{self, DeviceCtx, Event, Stream},
    host_region::HostRegion,
};

/// One device-resident KV unit (a TP shard's buffer, or an MLA replica's copy).
#[derive(Clone, Copy)]
pub struct Unit {
    pub device_id: u64,
    /// Device base pointer of the unit's buffer.
    pub ptr: usize,
    /// This unit's per-block byte stride (`KVCacheMemory.bytes_per_page` on
    /// the Python side, which built it from the caller's own buffer shapes).
    pub bytes: usize,
}

pub struct LeafMem {
    pub units: Vec<Unit>,
    pub replicated: bool,
}

#[derive(Clone)]
pub struct DeviceState {
    device_id: u64,
    ctx: DeviceCtx,
    /// One H2D aux stream per onload lane. Lane `i` uses `onload[i]` so two
    /// in-flight host hits do not enqueue on the same stream.
    onload: Vec<Stream>,
    offload: Stream,
    /// MAX's own compute stream for this device (not one of our aux streams):
    /// the gate records its event here, so the connector's copies wait on the
    /// forward pass rather than on themselves.
    compute_stream: Stream,
}

/// Compute-stream events (one per used device) recorded on the *main* thread in
/// `load`/`offload`; the lane's aux copies wait on them so a D2H reads only what
/// the forward wrote and an H2D does not race prior use of the destination — all
/// without touching MAX's compute stream from a lane thread. Each entry is
/// `(device_id, ctx, event)`: `device_id` keys the per-device wait, and `ctx` is
/// retained so [`Drop`] can destroy the event in its owning context. The gate is
/// the event's sole owner, so it destroys it exactly once when dropped (on
/// whichever lane thread finishes the transfer last) — without this the
/// driver-side event handles leak, one per device per batch.
pub struct ComputeGate {
    events: Vec<Event>,
    ctxs: Vec<DeviceCtx>,
}

impl ComputeGate {
    pub fn record(devices: &[DeviceState]) -> Self {
        let mut events = Vec::new();
        let mut ctxs = Vec::new();
        for device in devices {
            let ev = gpu::record_compute_event(&device.ctx, device.compute_stream);
            events.push(ev);
            ctxs.push(device.ctx);
        }
        ComputeGate { events, ctxs }
    }
    pub fn wait(&self, streams: &[Stream]) {
        for (stream, event) in streams.iter().zip_eq(self.events.iter()) {
            gpu::stream_wait_event(*stream, *event);
        }
    }
}

impl Drop for ComputeGate {
    fn drop(&mut self) {
        for (ctx, ev) in self.ctxs.iter().zip_eq(self.events.iter()) {
            gpu::destroy_event(ctx, *ev);
        }
    }
}

/// The per-replica copy engine.
pub struct CopyEngine {
    host: HostRegion,
    bytes_per_leaf: Vec<usize>,
    replicas: Vec<Vec<LeafMem>>,
    devices: Vec<Vec<DeviceState>>,
}

impl CopyEngine {
    /// Builds the engine, retaining a primary context + onload/offload streams
    /// per device and enabling peer access for every MLA peer pair.
    ///
    /// `host` is MAX's `_ChunkedStagingRegion`, which already page-locks /
    /// registers (or driver-allocates) every chunk for the device, so the
    /// engine does no host register of its own -- it just DMAs against it.
    ///
    /// `num_h2d_onload_lanes` is the number of concurrent H2D copies (and
    /// matching aux streams) per device. 1 is the historical single-copy FIFO.
    pub fn new(
        host: HostRegion,
        bytes_per_leaf: &[usize],
        replicas: Vec<Vec<LeafMem>>,
        device_ids: &[u64],
        device_to_stream: &HashMap<u64, u64>,
        num_h2d_onload_lanes: usize,
    ) -> Result<Self> {
        if num_h2d_onload_lanes == 0 {
            return Err(ConnectorError::Config(
                "num_h2d_onload_lanes must be at least 1".to_owned(),
            ));
        }
        // Construction runs on the caller's (init) thread and binds each device
        // to retain contexts / create streams / enable peer access; restore that
        // thread's context on return so we don't repoint it at our last device.
        let _ctx_guard = gpu::ContextGuard::save();

        let num_replicas = replicas.len();
        assert!(!device_ids.is_empty(), "must have at least one device");
        assert!(!replicas.is_empty(), "must have at least one replica");
        let devices_per_replica = device_ids.len() / num_replicas;

        let mut devices = Vec::new();
        for i in 0..num_replicas {
            let mut replica_devices = Vec::new();
            for j in 0..devices_per_replica {
                let did = device_ids[i * devices_per_replica + j];
                let ctx = gpu::retain_primary_context(did)?;
                let mut onload = Vec::with_capacity(num_h2d_onload_lanes);
                for _ in 0..num_h2d_onload_lanes {
                    onload.push(gpu::new_copy_stream(&ctx)?);
                }
                let offload = gpu::new_copy_stream(&ctx)?;
                let &compute_stream = device_to_stream.get(&did).ok_or_else(|| {
                    ConnectorError::Config(format!("no compute stream for device {did}"))
                })?;
                replica_devices.push(DeviceState {
                    device_id: did,
                    ctx,
                    onload,
                    offload,
                    compute_stream,
                });
            }
            devices.push(replica_devices);
        }

        // Enable peer access for MLA fan-out pairs.
        for replica_devices in &devices {
            for a in replica_devices {
                for b in replica_devices {
                    if a.device_id == b.device_id {
                        continue;
                    }
                    if gpu::can_access_peer(&a.ctx, &b.ctx).unwrap_or(false) {
                        drop(gpu::enable_peer_access(&a.ctx, &b.ctx));
                    }
                }
            }
        }

        let engine = CopyEngine {
            host,
            bytes_per_leaf: bytes_per_leaf.to_vec(),
            replicas,
            devices,
        };
        Ok(engine)
    }

    /// Records a compute-stream gate for `replica_idx` on the CURRENT thread
    /// (must be the scheduler thread). The lane's copies wait on it so a D2H
    /// reads only KV the forward wrote and an H2D does not race prior use of the
    /// destination — without a lane thread touching the compute stream.
    pub fn snapshot_compute(&self, replica_idx: usize) -> ComputeGate {
        ComputeGate::record(&self.devices[replica_idx])
    }

    /// H2D of `host_bid` -> device `dst_bid` for `replica_idx` (plus MLA p2p
    /// fan-out to peers), then **blocks the calling (lane) thread** until the
    /// copies land — no events, no polling. Called on onload lane `lane_idx`,
    /// so that lane keeps exactly one H2D in flight on its own stream
    /// (memcpy; sync; memcpy; sync ...). Other lanes can copy at the same
    /// time. Each stream still yields between its own blocks so one lane
    /// never monopolizes the copy engine against the forward's tiny I/O.
    pub fn h2d_sync(
        &self,
        replica_idx: usize,
        leaf_idx: usize,
        dst_bid: usize,
        host_bid: usize,
        lane_idx: usize,
        gate: &ComputeGate,
    ) {
        let leaf = &self.replicas[replica_idx][leaf_idx];
        let devices = &self.devices[replica_idx];
        let bytes = self.bytes_per_leaf[leaf_idx];
        assert!(
            devices
                .first()
                .is_some_and(|device| lane_idx < device.onload.len()),
            "onload lane {lane_idx} is out of range"
        );

        let all_streams = devices
            .iter()
            .map(|device| device.onload[lane_idx])
            .collect::<Vec<_>>();
        gate.wait(&all_streams);

        if leaf.replicated {
            let unit = leaf.units[0];
            let device = &devices[0];
            let stream = device.onload[lane_idx];

            assert_eq!(unit.bytes, bytes);

            let host_ptr = self.host.addr(host_bid * bytes, bytes);
            let dev_ptr = unit.ptr + dst_bid * bytes;

            gpu::memcpy_htod_async(&device.ctx, stream, dev_ptr as _, host_ptr as _, bytes);
            gpu::sync_stream(stream);

            for (peer_unit, peer_device) in leaf.units.iter().zip_eq(devices.iter()).skip(1) {
                let peer_stream = peer_device.onload[lane_idx];
                let peer_dev_ptr = peer_unit.ptr + dst_bid * bytes;

                gpu::memcpy_peer_async(
                    peer_stream,
                    peer_dev_ptr as _,
                    &peer_device.ctx,
                    dev_ptr as _,
                    &device.ctx,
                    bytes,
                );
            }

            for peer_device in devices.iter().skip(1) {
                gpu::sync_stream(peer_device.onload[lane_idx]);
            }
        } else {
            let mut offset = 0;
            for (unit, device) in leaf.units.iter().zip_eq(devices.iter()) {
                assert_eq!(unit.bytes * leaf.units.len(), bytes);

                let stream = device.onload[lane_idx];
                let host_ptr = self.host.addr(host_bid * bytes + offset, unit.bytes);
                let dev_ptr = unit.ptr + dst_bid * unit.bytes;

                gpu::memcpy_htod_async(
                    &device.ctx,
                    stream,
                    dev_ptr as _,
                    host_ptr as _,
                    unit.bytes,
                );

                offset += unit.bytes;
            }

            for device in devices.iter() {
                gpu::sync_stream(device.onload[lane_idx]);
            }
        }
    }

    /// D2H of device `src_bid` -> host `host_bid` for `replica_idx`, then
    /// **blocks the calling (lane) thread** until the copies land — no events,
    /// no polling. Called on the offload lane thread, so it keeps exactly one
    /// D2H in flight per offload stream (memcpy; sync; memcpy; sync ...). That
    /// idles the D2H engine between copies, so the connector's offload never
    /// monopolizes it against the forward pass's own D2H, and it drops the
    /// event-bundle machinery for the offload path.
    pub fn d2h_sync(
        &self,
        replica_idx: usize,
        leaf_idx: usize,
        host_bid: usize,
        src_bid: usize,
        gate: &ComputeGate,
    ) {
        let leaf = &self.replicas[replica_idx][leaf_idx];
        let devices = &self.devices[replica_idx];
        let bytes = self.bytes_per_leaf[leaf_idx];

        let all_streams = devices
            .iter()
            .map(|device| device.offload)
            .collect::<Vec<_>>();
        gate.wait(&all_streams);

        if leaf.replicated {
            assert!(bytes == leaf.units[0].bytes);

            let unit = leaf.units[0];
            let device = &devices[0];

            let stream = device.offload;
            let host_ptr = self.host.addr(host_bid * bytes, bytes);
            let dev_ptr = unit.ptr + src_bid * bytes;

            gpu::memcpy_dtoh_async(&device.ctx, stream, host_ptr as _, dev_ptr as _, bytes);
        } else {
            let mut offset = 0;
            for (unit, device) in leaf.units.iter().zip_eq(devices.iter()) {
                assert!(bytes == unit.bytes * leaf.units.len());

                let stream = device.offload;
                let host_ptr = self.host.addr(host_bid * bytes + offset, unit.bytes);
                let dev_ptr = unit.ptr + src_bid * unit.bytes;

                gpu::memcpy_dtoh_async(
                    &device.ctx,
                    stream,
                    host_ptr as _,
                    dev_ptr as _,
                    unit.bytes,
                );

                offset += unit.bytes;
            }
        };

        for device in devices.iter() {
            gpu::sync_stream(device.offload);
        }
    }

    pub fn host_ptr(&self, leaf_idx: usize, host_bid: usize) -> usize {
        let bytes = self.bytes_per_leaf[leaf_idx];
        self.host.addr(host_bid * bytes, bytes)
    }
}

impl Drop for CopyEngine {
    fn drop(&mut self) {
        // Drain in-flight copies before MAX frees the pinned host buffer.
        for replica_devices in &self.devices {
            for device in replica_devices {
                for stream in &device.onload {
                    gpu::sync_stream(*stream);
                }
                gpu::sync_stream(device.offload);
            }
        }
        // Release the GPU handles this engine acquired in `new`: the onload
        // aux streams, the offload stream, and the retained primary context.
        // Streams are destroyed only after the drain above.
        for replica_devices in &self.devices {
            for device in replica_devices {
                for stream in &device.onload {
                    gpu::destroy_stream(*stream);
                }
                gpu::destroy_stream(device.offload);
                gpu::release_primary_context(&device.ctx);
            }
        }
        // We neither registered nor own the host buffer: MAX's DeviceContext
        // fast pinned allocator page-locked it and unregisters + frees it, so
        // there is nothing to unregister here.
    }
}
