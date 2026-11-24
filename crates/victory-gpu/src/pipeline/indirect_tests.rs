use super::Pending;
use crate::draw::State;
use crate::host::{HostList, host, transition};
use crate::object::Object;
use crate::pipeline::{FUNCTIONS, Pipeline, host::HostPipeline};
use std::sync::{Arc, atomic::AtomicBool};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

const TRIANGLE: &str = r#"
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 p = id == 0 ? float2(-1,-1) : (id == 1 ? float2(-1,3) : float2(3,-1));
    return float4(p, 0.5, 1);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
float4 ps() : SV_Target { return float4(0,1,0,1); }
"#;

fn compile(source: &str, entry: PCSTR, target: PCSTR) -> Vec<u8> {
    let mut blob: Option<ID3DBlob> = None;
    let mut diagnostics: Option<ID3DBlob> = None;
    // SAFETY: input and output objects are live across the compile.
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("triangle"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut blob,
            Some(&mut diagnostics),
        )
        .unwrap();
        let blob = blob.unwrap();
        std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()).to_vec()
    }
}

// Only the two test shaders are supplied by the fixture. Root binding, indirect signatures,
// clamp, argument refresh, copy-back and queue submission run the port's implementation.
fn ready_triangle(device: &ID3D12Device) -> Box<Pipeline> {
    let vs = compile(TRIANGLE, s!("vs"), s!("vs_5_1"));
    let ps = compile(TRIANGLE, s!("ps"), s!("ps_5_1"));
    let bytecode = |b: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: b.as_ptr().cast(),
        BytecodeLength: b.len(),
    };
    // SAFETY: the blobs, root and pipeline description live through the calls.
    let (root, pso) = unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs).unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: bytecode(&vs),
            PS: bytecode(&ps),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
            D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let pso = device.CreateGraphicsPipelineState(&desc).unwrap();
        (root, pso)
    };
    ready(HostPipeline {
        compute: false,
        mesh: false,
        root,
        pso,
        stages: Vec::new(),
        slots: 0,
        inputs: Vec::new(),
        name: "audit triangle".to_owned(),
    })
}

fn ready(pipeline: HostPipeline) -> Box<Pipeline> {
    let p = Pending::new(Vec::new(), None);
    assert!(p.made.set(Some(pipeline)).is_ok());
    Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        shaders: Vec::new(),
        bound: AtomicBool::new(false),
        host: Arc::new(p),
    })
}

// Heap storage remains stable across moves; lists only use it within the fixture lifetime.
struct TitleBuffer(Box<[std::cell::UnsafeCell<u32>]>);

impl TitleBuffer {
    fn new(words: Vec<u32>) -> Self {
        let b = Self(words.into_iter().map(std::cell::UnsafeCell::new).collect());
        crate::shadow::register(b.at(0) as usize, b.at(0), (b.0.len() * 4) as u64);
        b
    }
    fn at(&self, word: usize) -> u64 {
        self.0[word].get() as u64
    }
}

impl Drop for TitleBuffer {
    fn drop(&mut self) {
        crate::shadow::unregister(self.at(0) as usize, self.at(0));
    }
}

// Host selection is process-wide. A machine without a hardware device
// can still run the same assertions on WARP in a fresh process.
fn have_adapter() -> bool {
    if host().is_some() {
        return true;
    }
    assert!(
        std::env::var_os("VICTORY_D3D12_WARP").is_none(),
        "WARP device unavailable"
    );
    let thread = std::thread::current();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", thread.name().unwrap(), "--nocapture"])
        .env("VICTORY_D3D12_WARP", "1")
        .status()
        .unwrap();
    assert!(result.success());
    false
}

fn render(
    ready: &Pipeline,
    producer: Option<&HostList>,
    record: impl FnOnce(&mut State, &HostList),
) -> (Vec<[u8; 4]>, D3D12_QUERY_DATA_PIPELINE_STATISTICS) {
    const SIZE: u32 = 4;
    let host = host().unwrap();
    let list = host.new_list().unwrap();
    let mut state = State::default();
    let fixture = (&ready.object) as *const Object;
    state.reset(fixture);
    state.set_topology(&list, 4);
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let texture_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: SIZE.into(),
        Height: SIZE,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut target: Option<ID3D12Resource> = None;
    // SAFETY: open list and live texture and viewport descriptions.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &texture_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some(&D3D12_CLEAR_VALUE {
                    Format: texture_desc.Format,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        Color: [0., 0., 0., 1.],
                    },
                }),
                &mut target,
            )
            .unwrap();
    }
    let target = target.unwrap();
    let rtv = host.rtv(&target, None);
    // SAFETY: a live RTV and an open list.
    unsafe {
        list.list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.list
            .ClearRenderTargetView(rtv, &[0., 0., 0., 1.], None);
        list.list.RSSetViewports(&[D3D12_VIEWPORT {
            Width: SIZE as f32,
            Height: SIZE as f32,
            MaxDepth: 1.,
            ..Default::default()
        }]);
        list.list.RSSetScissorRects(&[RECT {
            right: SIZE as i32,
            bottom: SIZE as i32,
            ..Default::default()
        }]);
    }
    let mut query: Option<ID3D12QueryHeap> = None;
    let query_desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
        Count: 1,
        NodeMask: 0,
    };
    let (stats, mapped) = host
        .shared_buffer(size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>() as u64)
        .unwrap();
    // SAFETY: the query and its destination outlive the submission.
    unsafe {
        host.device
            .CreateQueryHeap(&query_desc, &mut query)
            .unwrap();
    }
    let query = query.unwrap();
    // SAFETY: a pipeline statistics query on a direct list.
    unsafe {
        list.list
            .BeginQuery(&query, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
    }
    record(&mut state, &list);
    list.shadow_state(&stats, D3D12_RESOURCE_STATE_COPY_DEST);
    // SAFETY: the active query is ended before its result is copied.
    unsafe {
        list.list
            .EndQuery(&query, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list.list.ResolveQueryData(
            &query,
            D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
            0,
            1,
            &stats,
            0,
        );
    }
    // SAFETY: target is still in RENDER_TARGET; GPU is done before Map.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            &target,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )])
    };
    let (readback, footprint) =
        crate::capture::read_back(&host.device, &list.list, &target).unwrap();
    list.close();
    if let Some(producer) = producer {
        host.execute(&[producer, &list]);
    } else {
        host.execute(&[&list]);
    }
    let mut ptr = std::ptr::null_mut();
    let mut pixels = Vec::new();
    // SAFETY: the fence completed, and the footprint covers these pixels.
    unsafe {
        readback.Map(0, None, Some(&mut ptr)).unwrap();
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                let offset =
                    footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize + 4 * x;
                pixels.push(
                    ptr.cast::<u8>()
                        .add(offset)
                        .cast::<[u8; 4]>()
                        .read_unaligned(),
                );
            }
        }
        readback.Unmap(0, None);
    }
    // SAFETY: Host::execute waited for the fence before the CPU reads the query.
    let statistics = unsafe { mapped.cast::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>().read() };
    (pixels, statistics)
}

const GREEN: [u8; 4] = [0, 255, 0, 255];

#[test]
fn audit_render_indirect() {
    if !have_adapter() {
        return;
    }
    const RECORDS: usize = 2048;
    let fixture = ready_triangle(&host().unwrap().device);
    let untranslated = crate::pipeline::shader("vertex", b"fixture with no translatable shader");
    let mut rejected = std::ptr::null();
    assert_eq!(
        crate::pipeline::hand_out(
            vec![untranslated],
            Some(Box::default()),
            &crate::pipeline::PIPELINE_IIDS[0],
            &mut rejected
        ),
        crate::object::S_OK
    );
    assert!(crate::pipeline::host_pipeline(rejected).is_none());
    for (name, counted, populated) in [
        ("omit inactive batch", None, true),
        ("count=0, empty tail", Some(true), false),
        ("count=0, populated tail", Some(true), true),
        ("rejected pipeline, empty tail", Some(false), false),
        ("rejected pipeline, populated tail", Some(false), true),
    ] {
        let mut words = vec![0; 4 * (RECORDS + 1) + 1];
        if populated {
            words[..4 * RECORDS]
                .as_chunks_mut::<4>()
                .0
                .fill([512, 128, 0, 0]);
        }
        words[4 * RECORDS..4 * (RECORDS + 1)].copy_from_slice(&[3, 1, 0, 0]);
        let args = TitleBuffer::new(words);
        let (pixels, stats) = render(&fixture, None, |state, list| {
            let sig = (0, 16, RECORDS as u32);
            match counted {
                Some(true) => assert!(state.execute_indirect(
                    list,
                    sig,
                    args.at(0),
                    Some(args.at(4 * (RECORDS + 1)))
                )),
                Some(false) => {
                    state.set_pipeline(rejected);
                    assert!(!state.execute_indirect(list, sig, args.at(0), None));
                    state.set_pipeline(&fixture.object);
                }
                None => {}
            }
            assert!(state.execute_indirect(list, (0, 16, 1), args.at(4 * RECORDS), None));
        });
        let verts = stats.IAVertices;
        let green = pixels.iter().filter(|p| **p == GREEN).count();
        eprintln!("{name}: {green}/16 green, {verts} vertices");
        assert_eq!((green, verts), (16, 3), "{name}");
    }
    crate::pipeline::release(rejected);
}

/// A producer runs after the CPU refresh. Reading the count on the CPU would get zero.
fn gpu_store(list: &HostList, at: u64, value: u32) {
    const PRODUCER: &str = r#"
        RWByteAddressBuffer dest : register(u0);
        cbuffer Params : register(b0) { uint offset; uint value; };
        [RootSignature("UAV(u0), RootConstants(num32BitConstants=2,b0)")]
        [numthreads(1,1,1)] void main() { dest.Store(offset, value); }"#;
    let host = host().unwrap();
    let code = compile(PRODUCER, s!("main"), s!("cs_5_1"));
    let shadow = crate::shadow::find(at, 4).unwrap();
    shadow.refresh(list, at, 4);
    shadow.gpu_writes(list, at, 4);
    list.shadow_state(&shadow.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // SAFETY: live shader, root and destination. The list retains its bindings.
    unsafe {
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
        let pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .unwrap();
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list
            .SetComputeRootUnorderedAccessView(0, shadow.buffer.GetGPUVirtualAddress());
        let constants = [shadow.offset(at) as u32, value];
        list.list
            .SetComputeRoot32BitConstants(1, 2, constants.as_ptr().cast(), 0);
        list.list.Dispatch(1, 1, 1);
        list.keep(&root);
        list.keep(&pso);
    }
}

#[test]
fn counted_prefix_draws_exactly_the_live_records() {
    if !have_adapter() {
        return;
    }
    let fixture = ready_triangle(&host().unwrap().device);
    for (name, count, max, producer, separate, zero_tail, expected) in [
        ("partially filled", Some(2), 4, false, false, true, 9),
        (
            "live prefix with drawable tail",
            Some(1),
            4,
            false,
            true,
            false,
            3,
        ),
        (
            "GPU count in argument buffer",
            Some(2),
            4,
            true,
            false,
            false,
            9,
        ),
        (
            "GPU count in separate buffer",
            Some(3),
            4,
            true,
            true,
            false,
            21,
        ),
        ("count equals maximum", Some(4), 4, false, false, false, 45),
        ("count above maximum", Some(9), 2, false, false, false, 9),
        ("uncounted", None, 3, false, false, false, 21),
        ("zero maximum", Some(4), 0, false, false, false, 0),
    ] {
        // Count at +4 and arguments at +20. Padded 24-byte source stride,
        // with only the 16-byte payload present at the end of the buffer.
        let initial = if producer { 0 } else { count.unwrap_or(0) };
        let mut words = vec![0; 5 + 3 * 6 + 4];
        words[1] = initial;
        for i in 0..4 {
            if !zero_tail || i < 2 {
                words[5 + i * 6..9 + i * 6].copy_from_slice(&[3, 1 << i, 0, 0]);
            }
        }
        let args = TitleBuffer::new(words);
        let external = TitleBuffer::new(vec![0, 0, 0, initial]);
        let at = if separate { external.at(3) } else { args.at(1) };
        let (_, stats) = render(&fixture, None, |state, list| {
            if producer {
                gpu_store(list, at, count.unwrap());
            }
            assert!(state.execute_indirect(list, (0, 24, max), args.at(5), count.map(|_| at)));
        });
        let vertices = stats.IAVertices;
        eprintln!("{name}: {vertices} vertices");
        assert_eq!(vertices, expected, "{name}");
    }
}

fn ready_compute() -> Box<Pipeline> {
    let code = compile(
        r#"[RootSignature("")] [numthreads(2,1,1)] void main() {}"#,
        s!("main"),
        s!("cs_5_1"),
    );
    let device = &host().unwrap().device;
    // SAFETY: the shader and its embedded root signature live through creation.
    let (root, pso) = unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).unwrap();
        let pso = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .unwrap();
        (root, pso)
    };
    let shader = crate::pipeline::shader("compute", &code);
    shader.translated.get_or_init(|| {
        Some(crate::pipeline::Translated {
            host: victory_shader::emit::HostShader {
                stage: victory_shader::native::HwStage::Cs,
                dxil: code,
                user_data: vec![],
                samplers: vec![],
                static_samplers: vec![],
                root_cbvs: vec![],
                root_srvs: vec![],
            },
            used: vec![victory_shader::UsedResource {
                kind: victory_shader::ResourceType::Cbv,
                root: victory_shader::RootType::Table,
                slot: 0,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
            }],
        })
    });
    ready(HostPipeline {
        root,
        pso,
        compute: true,
        mesh: false,
        stages: vec![crate::pipeline::host::Stage {
            shader,
            user_data: None,
            root_cbvs: vec![],
            root_srvs: vec![],
        }],
        slots: 0,
        inputs: vec![],
        name: "audit compute".to_owned(),
    })
}

#[test]
fn dispatch_restores_the_title_compute_pipeline() {
    if !have_adapter() {
        return;
    }
    let pipeline = ready_compute();
    let args = TitleBuffer::new(vec![1, 3, 1, 1, 9, 1, 1]);
    let (_, stats) = render(&pipeline, None, |state, list| {
        assert!(state.execute_indirect(list, (2, 12, 2), args.at(1), Some(args.at(0))));
        state.dispatch(list, 1, 1, 1);
    });
    // One 64-thread clamp group, three indirect title groups, one direct title group.
    assert_eq!(stats.CSInvocations, 64 + (3 + 1) * 2);
}

#[test]
fn a_failed_descriptor_bind_leaves_the_next_draw_its_budget() {
    if !have_adapter() {
        return;
    }
    let triangle = ready_triangle(&host().unwrap().device);
    let compute = ready_compute();
    // A shader-visible view heap with one guest descriptor and no placement address.
    let desc = [0u32, 1, 1, 0, 0, 0, 0, 0];
    let mut handle = std::ptr::null();
    assert_eq!(
        crate::descriptor::create_heap(
            std::ptr::null(),
            desc.as_ptr().cast(),
            &crate::resource::PAGEABLE_IIDS[0],
            &mut handle
        ),
        crate::object::S_OK
    );
    let heap = crate::descriptor::heap(handle);
    let slots = heap.host().unwrap();
    while slots.dynamic(4096).is_some() {}
    while slots.dynamic(1).is_some() {}
    let dispatches = TitleBuffer::new([1024, 1, 1].repeat(2048));
    let draw = TitleBuffer::new(vec![3, 1, 0, 0]);
    let (pixels, stats) = render(&triangle, None, |state, list| {
        state.set_pipeline(&compute.object);
        state.set_heaps(&[handle]);
        state.set_table(true, 0, heap.start);
        assert!(!state.execute_indirect(list, (2, 12, 2048), dispatches.at(0), None));
        state.set_heaps(&[]);
        state.set_pipeline(&triangle.object);
        assert!(state.execute_indirect(list, (0, 16, 1), draw.at(0), None));
    });
    crate::object::release_boxed::<crate::descriptor::Heap>(handle);
    assert_eq!(pixels, [GREEN; 16]);
    assert_eq!(stats.IAVertices, 3);
    assert_eq!(stats.CSInvocations, 64); // only the successful draw's clamp
}

#[test]
fn a_prior_list_can_produce_both_arguments_and_count() {
    if !have_adapter() {
        return;
    }
    let host = host().unwrap();
    let fixture = ready_triangle(&host.device);
    let args = TitleBuffer::new(vec![0, 1, 0, 0, 3, 8, 0, 0]);
    let count = TitleBuffer::new(vec![0]);
    let producer = host.new_list().unwrap();
    gpu_store(&producer, args.at(0), 3);
    gpu_store(&producer, count.at(0), 1);
    producer.close();
    let (pixels, stats) = render(&fixture, Some(&producer), |state, list| {
        assert!(state.execute_indirect(list, (0, 16, 2), args.at(0), Some(count.at(0))));
    });
    assert_eq!(pixels, [GREEN; 16]);
    assert_eq!(stats.IAVertices, 3);
}

#[test]
fn switches_between_draw_indexed_compute_and_draw() {
    if !have_adapter() {
        return;
    }
    let triangle = ready_triangle(&host().unwrap().device);
    let compute = ready_compute();
    let draw = TitleBuffer::new(vec![3, 1, 0, 0]);
    let indexed = TitleBuffer::new(vec![3, 1, 1, (-1i32) as u32, 0]);
    let dispatch = TitleBuffer::new(vec![2, 1, 1]);
    let indices = TitleBuffer::new(vec![19, 1, 2, 3]);
    let view = D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: indices.at(0),
        SizeInBytes: 16,
        Format: DXGI_FORMAT_R32_UINT,
    };
    // SAFETY: both views are repr(C) with the same three integer fields in the same order.
    let view: crate::draw::IndexView = unsafe { std::mem::transmute(view) };
    let (pixels, stats) = render(&triangle, None, |state, list| {
        assert!(!state.execute_indirect(list, (1, 20, 1), indexed.at(0), None));
        state.set_index_buffer(Some(&view));
        assert!(state.execute_indirect(list, (1, 20, 1), indexed.at(0), None));
        state.set_pipeline(&compute.object);
        assert!(!state.execute_indirect(list, (0, 16, 1), draw.at(0), None));
        assert!(state.execute_indirect(list, (2, 12, 1), dispatch.at(0), None));
        state.set_pipeline(&triangle.object);
        assert!(state.execute_indirect(list, (0, 16, 1), draw.at(0), None));
        assert!(!state.execute_indirect(list, (10, 12, 1), dispatch.at(0), None));
    });
    assert_eq!(pixels, [GREEN; 16]);
    assert_eq!(stats.IAVertices, 6);
    assert_eq!(stats.CSInvocations, 3 * 64 + 4);
}

#[test]
fn parallel_lists_use_independent_counts_and_arguments() {
    if !have_adapter() {
        return;
    }
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        for active in [1, 2] {
            let barrier = &barrier;
            scope.spawn(move || {
                let fixture = ready_triangle(&host().unwrap().device);
                let args = TitleBuffer::new(vec![active, 3, 1, 0, 0, 3, 2, 0, 0]);
                let (pixels, stats) = render(&fixture, None, |state, list| {
                    assert!(state.execute_indirect(list, (0, 16, 2), args.at(1), Some(args.at(0))));
                    barrier.wait();
                });
                assert_eq!(pixels, [GREEN; 16]);
                assert_eq!(stats.IAVertices, if active == 1 { 3 } else { 9 });
            });
        }
    });
}
