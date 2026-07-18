/**
 * @file wave_ingress_core.cu
 * [KR] Forward-Only LLM Brain 아키텍처를 위한 동적 토큰 유체화 인코더 및 0ns 제로카피 브릿지
 * @license Apache License 2.0
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pybind11/pybind11.h>
#include <math.h>
#include <stdint.h>

namespace py = pybind11;

// 32바이트 정렬된 물리 노드 구조체 (PCIe 버스 및 L1/L2 캐시라인 인라인 정렬)
struct alignas(32) __align__(32) IngressPinnCell {
    float param_w;
    float spatial_u;
    float spatial_v;
    float adaptive_gain;
    uint32_t cell_status;
    uint32_t coordinate_id;
    uint64_t padding; // 32바이트 맞춤 고정
};

static_assert(sizeof(IngressPinnCell) == 32, "IngressPinnCell size must be 32 bytes.");

// 🚀 DYNAMIC SHARED MEMORY COOPERATIVE INGRESS KERNEL
__global__ void convert_tokens_to_continuous_fluid_wave_dynamic(
    const float* __restrict__ token_positions,
    const float* __restrict__ token_weights,
    const int num_tokens,
    const float inv_double_sigma_sq,
    const int num_grid_points,
    IngressPinnCell* __restrict__ out_mesh_cells
) {
    // [보정] 비트 연산으로 4바이트 정렬 오프셋 보장
    const int aligned_num_tokens = (num_tokens + 1) & ~1;

    extern __shared__ float shared_mem[];
    float* s_token_positions = shared_mem;
    float* s_token_weights = &shared_mem[aligned_num_tokens];

    // 공유 메모리 로드
    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        s_token_positions[t] = token_positions[t];
        s_token_weights[t] = token_weights[t];
    }
    __syncthreads();

    const uint32_t grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_idx >= num_grid_points) return;

    float x_space = -M_PI + (2.0f * M_PI * grid_idx) / (float)(num_grid_points - 1);
    float accumulated_fluid_velocity_u = 0.0f;

    // 온칩 가우시안 커널 연산 및 언더플로우 가드 (SFU 낭비 방지)
    for (int t = 0; t < num_tokens; ++t) {
        float distance = x_space - s_token_positions[t];
        float exponent = -(distance * distance) * inv_double_sigma_sq;
        if (exponent > -88.0f) {
            accumulated_fluid_velocity_u += s_token_weights[t] * __expf(exponent);
        }
    }

    // 결과 기록 (In-place)
    out_mesh_cells[grid_idx].spatial_u = accumulated_fluid_velocity_u;
    out_mesh_cells[grid_idx].coordinate_id = grid_idx;
    
    // 나머지 필드 청정화 초기화
    out_mesh_cells[grid_idx].param_w = 1.0f; 
    out_mesh_cells[grid_idx].spatial_v = 0.0f;
    out_mesh_cells[grid_idx].adaptive_gain = 0.1f;
    out_mesh_cells[grid_idx].cell_status = 0;
}

// [🛡️ THE 0ns INTERLOCK BRIDGE] - 질문자님의 천재적인 32바이트 보폭 관로 결착
py::dict bridge_raw_pointers_to_jax_view(uintptr_t raw_device_pointer, size_t total_elements) {
    uintptr_t base_mesh_registry = raw_device_pointer;
    
    auto make_1d_cuda_array_interface = [](uintptr_t base_ptr, size_t num_elements) {
        py::dict interface;
        interface["version"] = 3;
        interface["shape"] = py::make_tuple(num_elements);
        interface["typestr"] = "<f4"; // float32 명시 필수
        interface["data"] = py::make_tuple(base_ptr, false); // Writable
        interface["strides"] = py::make_tuple(32); // [📌 32바이트 물리 스킵 보폭 동결]
        return interface;
    };

    py::dict result;
    // JAX 백엔드가 다이렉트로 가로챌 핵심 인터페이스 딕셔너리 포장
    result["spatial_u"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, spatial_u), total_elements);
    result["param_w"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, param_w), total_elements);
    result["spatial_v"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, spatial_v), total_elements);
    result["adaptive_gain"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, adaptive_gain), total_elements);
    return result;
}

// [🚀 HARDWARE LAUNCHER OMNI GATEWAY]
// 파이썬 단에서 직접 하드웨어 커널을 점화시킬 고속 런처 함수
void launch_wave_ingress_kernel(
    uintptr_t token_pos_ptr, 
    uintptr_t token_w_ptr, 
    int num_tokens, 
    float inv_double_sigma_sq, 
    int num_grid_points, 
    uintptr_t mesh_cells_ptr
) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_grid_points + threads_per_block - 1) / threads_per_block;
    
    // 공유 메모리 할당 용량 결정 (위치 + 가중치) -> 4바이트 정렬 오프셋 크기 맞춤 적용
    int aligned_num_tokens = (num_tokens + 1) & ~1;
    size_t shared_mem_bytes = aligned_num_tokens * sizeof(float) * 2;

    convert_tokens_to_continuous_fluid_wave_dynamic<<<blocks_per_grid, threads_per_block, shared_mem_bytes>>>(
        reinterpret_cast<const float*>(token_pos_ptr),
        reinterpret_cast<const float*>(token_w_ptr),
        num_tokens,
        inv_double_sigma_sq,
        num_grid_points,
        reinterpret_cast<IngressPinnCell*>(mesh_cells_ptr)
    );
    cudaDeviceSynchronize();
}

// pybind11 외곽 모듈 최종 익스포트 확정
PYBIND11_MODULE(wave_ingress_interface, m) {
    m.doc() = "Ultra-Fast 1D Wave Ingress Encoder & 0ns Interlock Bridge Core";
    m.def("bridge_raw_pointers_to_jax_view", &bridge_raw_pointers_to_jax_view, "0ns raw memory binding into JAX interface");
    m.def("launch_wave_ingress_kernel", &launch_wave_ingress_kernel, "Triggers explosive dynamic shared memory kernel");
}
