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
//!
//! The host row can be longer than the units it holds: with direct disk I/O,
//! MAX pads every row to a whole number of 4 KiB so the disk tier can read and
//! write it in place. The copies move only the units' bytes; the padding goes
//! to disk and back but never reaches a device.

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
    /// How many of `units` hold distinct bytes: the first `shards`, of which
    /// unit `i` mirrors unit `i % shards`. 1 for a leaf replicated across the
    /// replica (an MLA latent), every unit for a sharded one, and a DCP
    /// group's size for a leaf that decode context parallelism splits by
    /// position within each of the replica's `units / shards` groups. A block
    /// moves to or from the host through the distinct units only, and an
    /// onload fans each one out to its mirrors.
    pub shards: usize,
}

/// One unit's copy of a block: its span of the block's host row, and the
/// device address of the block in that unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UnitSpan {
    host_offset: usize,
    dev_ptr: usize,
    len: usize,
}

impl LeafMem {
    /// A leaf whose every unit holds the whole block.
    #[must_use]
    pub fn replicated(units: Vec<Unit>) -> Self {
        Self { units, shards: 1 }
    }

    /// A leaf whose units hold distinct slices of the block.
    #[must_use]
    pub fn sharded(units: Vec<Unit>) -> Self {
        let shards = units.len();
        Self { units, shards }
    }

    /// The units that hold distinct bytes; the rest mirror them.
    fn distinct(&self) -> &[Unit] {
        &self.units[..self.shards.min(self.units.len())]
    }

    /// The bytes of one block the devices hold between their distinct units:
    /// a replicated leaf's one unit holds the whole block, a sharded leaf's
    /// units hold it between them.
    #[must_use]
    pub fn logical_row_bytes(&self) -> usize {
        self.distinct().iter().map(|unit| unit.bytes).sum()
    }

    /// Checks the leaf's units fit a host row of `stride` bytes.
    fn check_row(&self, leaf_idx: usize, stride: usize) -> Result<()> {
        let config = |msg: String| Err(ConnectorError::Config(msg));
        if self.units.is_empty() {
            return config(format!("leaf {leaf_idx} has no device units"));
        }
        if self.shards == 0 || !self.units.len().is_multiple_of(self.shards) {
            return config(format!(
                "leaf {leaf_idx}'s {} units do not split into mirrors of {} shards",
                self.units.len(),
                self.shards
            ));
        }
        if self
            .units
            .iter()
            .enumerate()
            .any(|(i, unit)| unit.bytes != self.units[i % self.shards].bytes)
        {
            return config(format!(
                "leaf {leaf_idx}'s mirrored units disagree on the block size"
            ));
        }
        let logical = self.logical_row_bytes();
        if logical == 0 || logical > stride {
            return config(format!(
                "leaf {leaf_idx}'s {logical}-byte block does not fit its {stride}-byte host row"
            ));
        }
        Ok(())
    }

    /// The copies that move block `dev_bid` to or from the host row of block
    /// `host_bid`, `stride` bytes apart, one per distinct unit. A replicated
    /// leaf copies through its first unit only, which then fans out to its
    /// peers; a mirrored one through its first `shards`. Each unit's device
    /// address steps by its own `bytes`, never by the (possibly padded)
    /// stride.
    fn spans(
        &self,
        stride: usize,
        host_bid: usize,
        dev_bid: usize,
    ) -> impl Iterator<Item = UnitSpan> + '_ {
        self.distinct()
            .iter()
            .scan(host_bid * stride, move |offset, unit| {
                let span = UnitSpan {
                    host_offset: *offset,
                    dev_ptr: unit.ptr + dev_bid * unit.bytes,
                    len: unit.bytes,
                };
                *offset += unit.bytes;
                Some(span)
            })
    }
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

        for leaves in &replicas {
            if leaves.len() != bytes_per_leaf.len() {
                return Err(ConnectorError::Config(format!(
                    "a replica has {} leaves, but bytes_per_leaf names {}",
                    leaves.len(),
                    bytes_per_leaf.len()
                )));
            }
            for (leaf_idx, (leaf, &stride)) in leaves.iter().zip(bytes_per_leaf).enumerate() {
                leaf.check_row(leaf_idx, stride)?;
                if leaf.units.len() != devices_per_replica {
                    return Err(ConnectorError::Config(format!(
                        "leaf {leaf_idx} has {} device units for {devices_per_replica} devices per replica",
                        leaf.units.len()
                    )));
                }
            }
        }

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
        let stride = self.bytes_per_leaf[leaf_idx];
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

        // Each distinct unit's slice from its own host span.
        for (span, device) in leaf.spans(stride, host_bid, dst_bid).zip(devices.iter()) {
            let stream = device.onload[lane_idx];
            let host_ptr = self.host.addr(span.host_offset, span.len);

            gpu::memcpy_htod_async(
                &device.ctx,
                stream,
                span.dev_ptr as _,
                host_ptr as _,
                span.len,
            );
        }
        let shards = leaf.distinct().len();
        for device in &devices[..shards] {
            gpu::sync_stream(device.onload[lane_idx]);
        }
        if shards == leaf.units.len() {
            return;
        }
        // Then each mirror from the distinct unit it copies, peer to peer: a
        // replicated leaf's first unit, or the same rank of the first DCP
        // group.
        for (i, (peer_unit, peer_device)) in leaf.units.iter().zip(devices.iter()).enumerate() {
            if i < shards {
                continue;
            }
            let (src_unit, src_device) = (&leaf.units[i % shards], &devices[i % shards]);
            gpu::memcpy_peer_async(
                peer_device.onload[lane_idx],
                (peer_unit.ptr + dst_bid * peer_unit.bytes) as _,
                &peer_device.ctx,
                (src_unit.ptr + dst_bid * src_unit.bytes) as _,
                &src_device.ctx,
                peer_unit.bytes,
            );
        }
        for peer_device in &devices[shards..] {
            gpu::sync_stream(peer_device.onload[lane_idx]);
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
        let stride = self.bytes_per_leaf[leaf_idx];

        let all_streams = devices
            .iter()
            .map(|device| device.offload)
            .collect::<Vec<_>>();
        gate.wait(&all_streams);

        // `spans` covers the distinct units alone, so a replicated leaf reads
        // the block from one device and a mirrored one from one DCP group.
        for (span, device) in leaf.spans(stride, host_bid, src_bid).zip(devices.iter()) {
            let host_ptr = self.host.addr(span.host_offset, span.len);

            gpu::memcpy_dtoh_async(
                &device.ctx,
                device.offload,
                host_ptr as _,
                span.dev_ptr as _,
                span.len,
            );
        }

        for device in devices.iter() {
            gpu::sync_stream(device.offload);
        }
    }

    /// The host address of `host_bid`'s whole row, padding included: the span
    /// the disk tier reads and writes.
    pub fn host_ptr(&self, leaf_idx: usize, host_bid: usize) -> usize {
        let bytes = self.bytes_per_leaf[leaf_idx];
        self.host.addr(host_bid * bytes, bytes)
    }

    /// The bytes of one `leaf_idx` block an H2D or D2H moves, without the host
    /// row's padding.
    #[must_use]
    pub fn logical_row_bytes(&self, leaf_idx: usize) -> usize {
        self.replicas[0][leaf_idx].logical_row_bytes()
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

#[cfg(test)]
mod tests {
    use super::{LeafMem, Unit, UnitSpan};

    fn unit(ptr: usize, bytes: usize) -> Unit {
        Unit {
            device_id: 0,
            ptr,
            bytes,
        }
    }

    #[test]
    fn a_padded_replicated_row_keeps_the_device_stride() {
        // DSv4.1's idx_c2a: 13056 bytes padded to a 16384-byte host row. The
        // device block must step by 13056, or the H2D lands in the next block.
        let leaf = LeafMem {
            units: vec![unit(0x1000_0000, 13056), unit(0x2000_0000, 13056)],
            shards: 1,
        };
        assert_eq!(leaf.logical_row_bytes(), 13056);
        let spans: Vec<UnitSpan> = leaf.spans(16384, 3, 5).collect();
        assert_eq!(
            spans,
            vec![UnitSpan {
                host_offset: 3 * 16384,
                dev_ptr: 0x1000_0000 + 5 * 13056,
                len: 13056,
            }]
        );
    }

    #[test]
    fn a_padded_sharded_row_packs_its_units_and_pads_the_tail() {
        let leaf = LeafMem {
            units: vec![unit(0x1000_0000, 4352), unit(0x2000_0000, 4352)],
            shards: 2,
        };
        assert_eq!(leaf.logical_row_bytes(), 8704);
        let spans: Vec<UnitSpan> = leaf.spans(12288, 2, 7).collect();
        assert_eq!(
            spans,
            vec![
                UnitSpan {
                    host_offset: 2 * 12288,
                    dev_ptr: 0x1000_0000 + 7 * 4352,
                    len: 4352,
                },
                UnitSpan {
                    host_offset: 2 * 12288 + 4352,
                    dev_ptr: 0x2000_0000 + 7 * 4352,
                    len: 4352,
                },
            ]
        );
    }

    #[test]
    fn an_unpadded_row_is_unchanged() {
        let leaf = LeafMem {
            units: vec![unit(0x1000_0000, 8192)],
            shards: 1,
        };
        let spans: Vec<UnitSpan> = leaf.spans(8192, 4, 4).collect();
        assert_eq!(
            spans,
            vec![UnitSpan {
                host_offset: 4 * 8192,
                dev_ptr: 0x1000_0000 + 4 * 8192,
                len: 8192,
            }]
        );
    }

    #[test]
    fn a_row_its_units_overflow_is_refused() {
        let sharded = LeafMem::sharded(vec![unit(0, 4096), unit(0, 4096)]);
        sharded.check_row(0, 8192).expect("an exact fit is fine");
        sharded.check_row(0, 12288).expect("padding is fine");
        let err = sharded.check_row(1, 4096).unwrap_err().to_string();
        assert!(err.contains("leaf 1"), "{err}");

        let mismatched = LeafMem::replicated(vec![unit(0, 4096), unit(0, 2048)]);
        assert!(mismatched.check_row(0, 4096).is_err());

        let empty = LeafMem::sharded(Vec::new());
        assert!(empty.check_row(0, 4096).is_err());

        // Three units cannot mirror two shards.
        let ragged = LeafMem {
            units: vec![unit(0, 4096); 3],
            shards: 2,
        };
        assert!(ragged.check_row(0, 8192).is_err());
    }

    #[test]
    fn a_mirrored_row_moves_through_its_first_group_only() {
        // TP4 x DCP2: two groups of two ranks, each group a copy of the
        // block. The host row holds one group's two slices; the second
        // group's units mirror the first's.
        let leaf = LeafMem {
            units: vec![
                unit(0x1000_0000, 4096),
                unit(0x2000_0000, 4096),
                unit(0x3000_0000, 4096),
                unit(0x4000_0000, 4096),
            ],
            shards: 2,
        };
        leaf.check_row(0, 8192).expect("one group fills the row");
        assert!(leaf.check_row(0, 4096).is_err());
        assert_eq!(leaf.logical_row_bytes(), 8192);
        let spans: Vec<UnitSpan> = leaf.spans(8192, 1, 3).collect();
        assert_eq!(
            spans,
            vec![
                UnitSpan {
                    host_offset: 8192,
                    dev_ptr: 0x1000_0000 + 3 * 4096,
                    len: 4096,
                },
                UnitSpan {
                    host_offset: 8192 + 4096,
                    dev_ptr: 0x2000_0000 + 3 * 4096,
                    len: 4096,
                },
            ]
        );
    }
}
