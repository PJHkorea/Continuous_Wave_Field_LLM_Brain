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
    // [파트 1-A: 32비트 부동소수점 수학 필드군 - 총 16바이트]
    float param_w;          // 가중치 레지스터 변위
    float spatial_u;        // 1D 유체화 속도장 (U)
    float spatial_v;        // 1D 유체화 속도장 (V)
    float adaptive_gain;    // 적응형 이득 변수

    // [파트 1-B: 32비트 상태 및 기하학 제어 필드군 - 총 8바이트]
    uint32_t cell_status;   // 무분기 방화벽 제어 상태 비트
    uint32_t coordinate_id; // 공간 격자점 고유 바인딩 인덱스

    // [파트 1-C: 명시적 버스 정렬 패딩 - 총 8바이트]
    // 4바이트 변수 2개 뒤에 4바이트 패딩 2개를 붙여 정렬 경계를 수동으로 완전 동결
    uint32_t hardware_pad1; 
    uint32_t hardware_pad2;
};

// 실리콘 레벨에서 캐시라인 파편화 및 뱅크 스톨(Bank Stall)이 일어나는지 컴파일 시점에 완벽 가드
static_assert(sizeof(IngressPinnCell) == 32, "IngressPinnCell size must be exactly 32 bytes for JAX stride synchronization.");


// 🚀 DYNAMIC SHARED MEMORY COOPERATIVE INGRESS KERNEL
__global__ void convert_tokens_to_continuous_fluid_wave_dynamic(
    const float* __restrict__ token_positions,
    const float* __restrict__ token_weights,
    const int num_tokens,
    const float inv_double_sigma_sq,
    const int num_grid_points,
    IngressPinnCell* __restrict__ out_mesh_cells
) {
    // [보정] 32바이트 하드웨어 대역폭(8개 float 뱅크) 단위로 완전히 정렬하여 공유 메모리 뱅크 스톨 박멸
    const int aligned_num_tokens = (num_tokens + 7) & ~7;

    extern __shared__ float shared_mem[];
    float* s_token_positions = shared_mem;
    float* s_token_weights = &shared_mem[aligned_num_tokens]; // 완전한 32바이트 뱅크 정렬 오프셋 확보

    // 1. [COOPERATIVE LOADING] 온칩 메모리 고속 병렬 이송
    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        s_token_positions[t] = token_positions[t];
        s_token_weights[t] = token_weights[t];
    }
    __syncthreads();

    const uint32_t grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_idx >= num_grid_points) return;

    // 2. [ON-CHIP EXPLOSIVE LOOP]
    float x_space = -M_PI + (2.0f * M_PI * grid_idx) / (float)(num_grid_points - 1);
    float accumulated_fluid_velocity_u = 0.0f;

    for (int t = 0; t < num_tokens; ++t) {
        float distance = x_space - s_token_positions[t];
        float exponent = -(distance * distance) * inv_double_sigma_sq;
        
        // 하드웨어 언더플로우 하한선(-87.3f) 감안 가드로 SFU 파이프라인 지연 최소화
        if (exponent > -88.0f) {
            accumulated_fluid_velocity_u += s_token_weights[t] * __expf(exponent);
        }
    }

    // 3. [REAL-SPACE DATA HARD COMMIT] 구조체 최적화 레이아웃에 맞추어 인라인 기록
    out_mesh_cells[grid_idx].spatial_u = accumulated_fluid_velocity_u;
    out_mesh_cells[grid_idx].coordinate_id = grid_idx;
    
    // 나머지 동역학 제어 필드 청정화 및 리셋 초기화
    out_mesh_cells[grid_idx].param_w = 1.0f; 
    out_mesh_cells[grid_idx].spatial_v = 0.0f;
    out_mesh_cells[grid_idx].adaptive_gain = 0.1f;
    out_mesh_cells[grid_idx].cell_status = 0;
}

// [🛡️ THE 0ns INTERLOCK BRIDGE] - 완전 무결한 6대 물리 제어선 관로 결착
py::dict bridge_raw_pointers_to_jax_view(uintptr_t raw_device_pointer, size_t total_elements) {
    uintptr_t base_mesh_registry = raw_device_pointer;
    
    // 기본 배열 인터페이스 빌더 (데이터 타입 유연 대응 구조)
    auto make_1d_cuda_array_interface = [](uintptr_t base_ptr, size_t num_elements, const char* typestr) {
        py::dict interface;
        interface["version"] = 3;
        interface["shape"] = py::make_tuple(num_elements);
        interface["typestr"] = typestr; // 데이터 레이아웃에 따른 동적 배정 ("<f4" 또는 "<u4")
        interface["data"] = py::make_tuple(base_ptr, false); // Writable
        interface["strides"] = py::make_tuple(32); // 32바이트 물리 스킵 보폭 하드 로킹
        return interface;
    };

    py::dict result;
    
    // [📌 파트 2-A: JAX/XLA가 하이재킹할 float32 수학 필드군 전사]
    result["param_w"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, param_w), total_elements, "<f4");
    result["spatial_u"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, spatial_u), total_elements, "<f4");
    result["spatial_v"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, spatial_v), total_elements, "<f4");
    result["adaptive_gain"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, adaptive_gain), total_elements, "<f4");
    
    // [📌 파트 2-B: JAX 백엔드 메모리 구조에 완벽 대응하는 uint32 제어 필드군 추가 결착]
    result["cell_status"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, cell_status), total_elements, "<u4");
    result["coordinate_id"] = make_1d_cuda_array_interface(base_mesh_registry + offsetof(IngressPinnCell, coordinate_id), total_elements, "<u4");
    
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
    
    # // [보정] 2단계 커널 내부의 32바이트 하드웨어 대역폭(8개 float 뱅크) 정렬식과 1:1 싱크 매칭
    int aligned_num_tokens = (num_tokens + 7) & ~7;
    size_t shared_mem_bytes = aligned_num_tokens * sizeof(float) * 2;

    # // 0ns 완전 비동기 사격 실행 - 커널 명령을 가속기 명령어 큐(Stream 0)에 던진 후 즉시 호스트 리턴
    convert_tokens_to_continuous_fluid_wave_dynamic<<<blocks_per_grid, threads_per_block, shared_mem_bytes>>>(
        reinterpret_cast<const float*>(token_pos_ptr),
        reinterpret_cast<const float*>(token_w_ptr),
        num_tokens,
        inv_double_sigma_sq,
        num_grid_points,
        reinterpret_cast<IngressPinnCell*>(mesh_cells_ptr)
    );
    
    # // [🛡️ PIPELINE SYNC UNLEASHED] CPU 블로킹 유발 지점을 영구 청산하여 JAX 비동기 연산과 결착 유도
    # // cudaDeviceSynchronize(); -> 삭제 완료
}

// pybind11 외곽 모듈 최종 익스포트 확정
PYBIND11_MODULE(wave_ingress_interface, m) {
    m.doc() = "Ultra-Fast 1D Wave Ingress Encoder & 0ns Interlock Bridge Core";
    m.def("bridge_raw_pointers_to_jax_view", &bridge_raw_pointers_to_jax_view, "0ns raw memory binding into JAX interface");
    m.def("launch_wave_ingress_kernel", &launch_wave_ingress_kernel, "Triggers explosive dynamic shared memory kernel");
}

