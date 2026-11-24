//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. A compute shader copies the active records
//! into a private buffer, clamps them, and saves the count that selects them.
//! ExecuteIndirect consumes that snapshot on the same list.
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
// GPU-produced inputs may return to a writable state between bindings.
#define RS "RootConstants(num32BitConstants=9, b0), SRV(t0, flags=DATA_VOLATILE), SRV(t1, flags=DATA_VOLATILE), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind, stride, maximum, first, window, src_offset, dst_offset, counted, count_offset;
};
ByteAddressBuffer source : register(t0);
ByteAddressBuffer counter : register(t1);
RWByteAddressBuffer output_args : register(u0);
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;

// One group owns this window's budget. All lanes load and store in parallel;
// the group leader admits records in command order, without cross-group atomics.
groupshared uint live;
groupshared uint remaining;
groupshared uint costs[64];

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    if (lane == 0) {
        uint active = counted ? min(maximum, counter.Load(count_offset)) : maximum;
        if (first == 0) output_args.Store(0, active);
        live = first < active ? min(window, active - first) : 0;
        remaining = BUDGET - min(budget.Load(0), BUDGET);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint tile = 0; tile < live; tile += 64) {
        uint index = tile + lane;
        uint4 args = 0;
        uint start_instance = 0;
        uint cost = 0;
        if (index < live) {
            uint at = src_offset + index * stride;
            if (kind <= 1) {
                args = source.Load4(at);
                if (kind == 1) start_instance = source.Load(at + 16);
                args.x = min(args.x, 1u << 24);
                args.y = min(args.y, (1u << 26) / max(args.x, 1u));
                cost = args.x * args.y;
            } else {
                args.xyz = min(source.Load3(at), uint3(65535, 65535, 65535));
                if (kind == 2) {
                    if (args.z != 0 && args.x * args.y > GROUPS / args.z) args.xyz = 0;
                } else {
                    args.yz = min(args.yz, uint2(1, 1));
                }
                cost = args.x * args.y * args.z * (kind == 2 ? 64u : 128u);
            }
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            uint left = remaining;
            for (uint i = 0; i < min(64u, live - tile); ++i) {
                if (costs[i] <= left) left -= costs[i];
                else costs[i] = 0xffffffff;
            }
            remaining = left;
        }
        GroupMemoryBarrierWithGroupSync();
        if (index < live) {
            if (costs[lane] == 0xffffffff) args.x = 0;
            if (kind <= 1) {
                uint dst = dst_offset + index * (kind == 0 ? 16u : 20u);
                output_args.Store4(dst, args);
                if (kind == 1) output_args.Store(dst + 16, start_instance);
            } else {
                output_args.Store3(dst_offset + index * 12, args.xyz);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0 && live != 0) budget.Store(0, BUDGET - remaining);
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Only the signature payload is interpreted; the title stride can include padding.
#[derive(Clone, Copy)]
pub(crate) struct Layout {
    kind: u32,
    stride: u32,
    maximum: u32,
    pub(crate) width: u32,
}

impl Layout {
    pub(crate) fn new(kind: u32, stride: u32, maximum: u32) -> Option<Self> {
        let width = match kind {
            0 => 16,
            1 => 20,
            2 | 10 => 12,
            _ => return None,
        };
        (stride >= width && stride.is_multiple_of(4)).then_some(Self {
            kind,
            stride,
            maximum,
            width,
        })
    }

    pub(crate) fn span(self) -> u64 {
        if self.maximum == 0 {
            return 0;
        }
        u64::from(self.maximum - 1) * u64::from(self.stride) + u64::from(self.width)
    }
}

// Space for the count, then packed records. Root SRV/UAV addresses are
// 16-byte aligned even when a guest buffer offset is only dword aligned.
const HEADER: u64 = 16;
const WINDOW_RECORDS: u32 = 4096;
pub(crate) struct Prepared {
    clamp: &'static Clamp,
    budget: u64,
    scratch: ID3D12Resource,
}

pub(crate) fn prepare(
    list: &HostList,
    layout: Layout,
    args: u64,
    count: Option<u64>,
) -> Option<Prepared> {
    if !args.is_multiple_of(4) || count.is_some_and(|at| !at.is_multiple_of(4)) {
        return None;
    }
    let clamp = clamp()?;
    let size = HEADER + u64::from(layout.maximum) * u64::from(layout.width);
    let mut slot = list
        .indirect_scratch
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if slot.as_ref().is_none_or(|buffer| {
        // SAFETY: the list keeps the cached buffer alive.
        unsafe { buffer.GetDesc().Width < size }
    }) {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.next_multiple_of(64 << 10),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut buffer = None;
        // SAFETY: valid descriptions and an owned out pointer.
        unsafe {
            host()?
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .ok()?;
        }
        if let Some(old) = slot.replace(buffer?) {
            list.keep(&old);
        }
    }
    Some(Prepared {
        clamp,
        budget: list.budget()?,
        scratch: slot.as_ref()?.clone(),
    })
}

impl Prepared {
    pub(crate) fn record(
        &self,
        list: &HostList,
        layout: Layout,
        (args, address): (&ID3D12Resource, u64),
        count: Option<(&ID3D12Resource, u64)>,
    ) {
        let read = D3D12_RESOURCE_STATE_GENERIC_READ;
        let was = list.shadow_state(args, read);
        let other = count
            .filter(|(b, _)| *b != args)
            .map(|(b, _)| (b, list.shadow_state(b, read)));
        let counter = count.map_or(address, |(_, at)| at);
        let l = &list.list;
        // SAFETY: the list owns the scratch and budget; refresh holds both input shadows.
        unsafe {
            let base = self.scratch.GetGPUVirtualAddress();
            l.ResourceBarrier(&[
                uav_barrier(),
                transition(
                    &self.scratch,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
            l.SetComputeRootSignature(&self.clamp.root);
            l.SetPipelineState(&self.clamp.pso);
            l.SetComputeRootShaderResourceView(2, counter & !15);
            l.SetComputeRootUnorderedAccessView(4, self.budget);
            // Rebase each window so the shader's byte offsets stay 32-bit and
            // each workgroup has a bounded number of records.
            let mut first = 0;
            while first < layout.maximum {
                let src = address + u64::from(first) * u64::from(layout.stride);
                let dst = HEADER + u64::from(first) * u64::from(layout.width);
                let dst_base = if first == 0 { 0 } else { dst & !15 };
                let fit = 1 + (u32::MAX - (src & 15) as u32 - layout.width) / layout.stride;
                let window = (layout.maximum - first).min(WINDOW_RECORDS).min(fit);
                let constants = [
                    layout.kind,
                    layout.stride,
                    layout.maximum,
                    first,
                    window,
                    (src & 15) as u32,
                    (dst - dst_base) as u32,
                    u32::from(count.is_some()),
                    (counter & 15) as u32,
                ];
                l.SetComputeRoot32BitConstants(0, 9, constants.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, src & !15);
                l.SetComputeRootUnorderedAccessView(3, base + dst_base);
                if first != 0 {
                    // Successive windows read the preceding window's updated work tally.
                    l.ResourceBarrier(&[uav_barrier()]);
                }
                l.Dispatch(1, 1, 1);
                first += window;
            }
            l.ResourceBarrier(&[transition(
                &self.scratch,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
        // The title pipeline can also bind an input shadow for writes.
        list.shadow_state(args, was);
        if let Some((buffer, prior)) = other {
            list.shadow_state(buffer, prior);
        }
    }

    pub(crate) fn execute(
        &self,
        list: &HostList,
        signature: &ID3D12CommandSignature,
        maximum: u32,
    ) {
        // SAFETY: the private count and arguments were produced together and both are indirect-readable.
        unsafe {
            list.list
                .ExecuteIndirect(signature, maximum, &self.scratch, HEADER, &self.scratch, 0);
            list.list.ResourceBarrier(&[transition(
                &self.scratch,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn layout_ends_at_the_last_payload() {
        use super::Layout;
        assert_eq!(Layout::new(0, 24, 0).unwrap().span(), 0);
        assert_eq!(Layout::new(0, 24, 4).unwrap().span(), 88);
        assert_eq!(
            Layout::new(1, 20, u32::MAX).unwrap().span(),
            20 * u64::from(u32::MAX)
        );
        assert!(Layout::new(1, 16, 1).is_none());
        assert!(Layout::new(2, 13, 1).is_none());
        assert!(Layout::new(99, 16, 1).is_none());
        let stride = u32::MAX - 3;
        assert_eq!(
            Layout::new(2, stride, u32::MAX).unwrap().span(),
            (u64::from(u32::MAX) - 1) * u64::from(stride) + 12
        );
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn packed_arguments_preserve_the_source_and_budget_on_replay() {
        use super::*;
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        // These records only go through the clamp. Read back its output without issuing their work.
        // After the first large draw fits, the second is over budget; a small draw still fits.
        const LARGE: [u32; 5] = [1 << 24, 4, 0, 0, 0];
        const CASES: [(u32, [u32; 5], [u32; 5]); 9] = [
            (
                1,
                [3, 2, 7, (-3i32) as u32, 11],
                [3, 2, 7, (-3i32) as u32, 11],
            ),
            (2, [2, 3, 4, 0, 0], [2, 3, 4, 0, 0]),
            (10, [2, 3, 4, 0, 0], [2, 1, 1, 0, 0]),
            (0, LARGE, LARGE),
            (0, LARGE, [0, 4, 0, 0, 0]),
            (0, [3, 1, 0, 0, 0], [3, 1, 0, 0, 0]),
            (2, [u32::MAX, u32::MAX, u32::MAX, 0, 0], [0; 5]),
            (2, [65535, 65535, 0, 0, 0], [65535, 65535, 0, 0, 0]),
            (10, [u32::MAX, u32::MAX, 0, 0, 0], [65535, 1, 0, 0, 0]),
        ];
        let bytes = (CASES.len() * size_of::<[u32; 5]>()) as u64;
        let (source, input) = host.shared_buffer(bytes).unwrap();
        let (readback, output) = host.shared_buffer(bytes).unwrap();
        list.shadow_state(&readback, D3D12_RESOURCE_STATE_COPY_DEST);
        for (i, &(kind, words, _)) in CASES.iter().enumerate() {
            let offset = (i * size_of::<[u32; 5]>()) as u64;
            // SAFETY: the map has one five-dword slot per case; no GPU work is in flight.
            let address = unsafe {
                input.cast::<[u32; 5]>().add(i).write(words);
                source.GetGPUVirtualAddress() + offset
            };
            let layout = Layout::new(kind, 20, 1).unwrap();
            let prepared = prepare(&list, layout, address, None).unwrap();
            prepared.record(&list, layout, (&source, address), None);
            // SAFETY: copy only the packed payload, then return the scratch to its reusable state.
            unsafe {
                list.list.ResourceBarrier(&[transition(
                    &prepared.scratch,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.list.CopyBufferRegion(
                    &readback,
                    offset,
                    &prepared.scratch,
                    HEADER,
                    u64::from(layout.width),
                );
                list.list.ResourceBarrier(&[transition(
                    &prepared.scratch,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
        list.close();
        for _ in 0..2 {
            host.execute(&[&list]);
            for (i, &(kind, before, expected)) in CASES.iter().enumerate() {
                let words = Layout::new(kind, 20, 1).unwrap().width as usize / 4;
                // SAFETY: the submission has completed; each range is within its mapped buffer.
                unsafe {
                    assert_eq!(input.cast::<[u32; 5]>().add(i).read(), before);
                    assert_eq!(
                        std::slice::from_raw_parts(output.cast::<u32>().add(i * 5), words),
                        &expected[..words]
                    );
                }
            }
        }
    }

    #[test]
    fn ordered_budget_and_count_across_a_window_boundary() {
        use super::*;
        let Some(host) = host() else { return };
        const RECORDS: usize = 4101;
        const STRIDE: usize = 6;
        let bytes = (4 + (RECORDS - 1) * STRIDE + 4) * 4;
        let list = host.new_list().unwrap();
        let (source, src) = host.shared_buffer(bytes as u64).unwrap();
        let total = HEADER as usize + RECORDS * 16;
        let (readback, dst) = host.shared_buffer(total as u64).unwrap();
        // Two safe, unexecuted inputs leave room for 128 single-vertex records.
        // Each record has an observable ID, including the first record of the second window.
        let mut initial = vec![0xdeadbeef; bytes / 4];
        initial[1] = u32::MAX;
        for i in 0..RECORDS {
            let (x, y) = match i {
                0 => (1 << 24, 4),
                1 => ((1 << 24) - 32, 4),
                2 => (129, 1),
                3..=129 => (1, 1),
                130..=4095 => (0, 1),
                _ => (1, 1),
            };
            initial[4 + i * STRIDE..8 + i * STRIDE].copy_from_slice(&[x, y, i as u32, 93]);
        }
        // SAFETY: one mapped allocation of exactly bytes, no work in flight.
        let base = unsafe {
            std::ptr::copy_nonoverlapping(initial.as_ptr(), src.cast::<u32>(), initial.len());
            source.GetGPUVirtualAddress()
        };
        let layout = Layout::new(0, 24, RECORDS as u32).unwrap();
        assert!(prepare(&list, layout, base + 2, None).is_none());
        assert!(prepare(&list, layout, base + 16, Some(base + 2)).is_none());
        let prepared = prepare(&list, layout, base + 16, Some(base + 4)).unwrap();
        prepared.record(
            &list,
            layout,
            (&source, base + 16),
            Some((&source, base + 4)),
        );
        list.shadow_state(&readback, D3D12_RESOURCE_STATE_COPY_DEST);
        // SAFETY: both allocations cover total, and the list owns them through the fence.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &prepared.scratch,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list
                .CopyBufferRegion(&readback, 0, &prepared.scratch, 0, total as u64);
            list.list.ResourceBarrier(&[transition(
                &prepared.scratch,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        list.close();
        let check = |count: usize| {
            // SAFETY: all host submissions have finished. Only initialized dwords are read.
            unsafe {
                assert_eq!(dst.cast::<u32>().read(), count as u32);
                for i in 0..count {
                    let mut expected: [u32; 4] =
                        initial[4 + i * STRIDE..8 + i * STRIDE].try_into().unwrap();
                    if i == 2 || i >= 4097 {
                        expected[0] = 0;
                    }
                    assert_eq!(
                        dst.add(HEADER as usize).cast::<[u32; 4]>().add(i).read(),
                        expected,
                        "record {i}"
                    );
                }
                assert_eq!(
                    std::slice::from_raw_parts(src.cast::<u32>().add(4), initial.len() - 4),
                    &initial[4..]
                );
            }
        };
        for count in [u32::MAX, 4097, 0, 1, u32::MAX] {
            // SAFETY: the previous submission completed before this CPU write.
            unsafe {
                src.cast::<u32>().add(1).write(count);
            }
            host.execute(&[&list]);
            check((count as usize).min(RECORDS));
        }
        host.execute(&[&list, &list]);
        check(RECORDS);
    }
}
