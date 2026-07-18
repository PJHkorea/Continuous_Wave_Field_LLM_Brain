# wave_frontend_bridge.py
# [KR] C++ CUDA 물리 포인터와 JAX AI 공간을 연동하는 0ns 제로카피 무복사 프론트엔드 브릿지 관로
# License: Apache License 2.0

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

try:
    # 빌드 및 컴파일이 완공된 C++ pybind11 확장 모듈 수입
    import wave_ingress_interface 
except ImportError as framework_fault:
    raise ImportError(
        "[🚨 wave_frontend_bridge CRITICAL FAULT] C++ pybind11 module 'wave_ingress_interface' not found. "
        "Please execute nvcc compilation and check your PYTHONPATH baseline."
    ) from framework_fault


class CUDAInterfaceBridge:
    """
    [⛓ JAX ARRAY ADAPTER CAPSULE]
    C++ 단에서 __cuda_array_interface__ 규격으로 패키징해 준 32바이트 보폭의 
    VRAM 물리 주소 사전(Dictionary)을 JAX/XLA 백엔드가 합법적으로 가로채도록 속이는 프로토콜 팩토리입니다.
    """
    def __init__(self, interface_dict: dict) -> None:
        if not isinstance(interface_dict, dict):
            raise TypeError("[🚨 CUDAInterfaceBridge] Input interface must be a standard python dictionary structure.")

        # [🎯 FLEXIBLE DUCT ENGAGEMENT] 중첩 딕셔너리와 원시 명세 딕셔너리 양방향 유연 대응
        # 인스턴스 캡슐 내부에 원본 사전 데이터를 안전하게 격리 보관
        self._raw_interface: Dict[str, Any] = interface_dict.get("__cuda_array_interface__", interface_dict)
        
        # [🛡 SILICON VALIDITY GUARD] JAX/XLA가 하이재킹할 필수 3대 원소("data", "shape", "typestr") 유효성 최종 확인
        assert all(key in self._raw_interface for key in ("data", "shape", "typestr")), (
            f"[🚨 CUDAInterfaceBridge] Invalid CUDA Array Interface layout. "
            f"The hardware-level pointer constraints ('data', 'shape', 'typestr') keys must be defined. "
            f"Current keys: {list(self._raw_interface.keys())}"
        )

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        """[👑 PROTOCOL EXPANSION DUCT] JAX/XLA 백엔드가 어댑터를 표준 외곽선으로 오인하고 침투하도록 허용하는 관로"""
        return self._raw_interface


def fluidic_token_frontend_encoder(
    token_positions_ptr: int,
    token_weights_ptr: int,
    num_tokens: int,
    num_grid_points: int,
    mesh_cells_ptr: int
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """C++ 하드웨어 런처를 격발하고 6대 필드를 JAX 텐서로 제로카피 변환"""
    
    # 1. 가우시안 소산 대역폭 사전 계산
    inv_double_sigma_sq = float(1.0 / (2.0 * (0.35 ** 2)))
    
    # 2. C++ 커널 가동 (6대 필드 주소선 수집)
    wave_ingress_interface.launch_wave_ingress_kernel(
        token_positions_ptr, token_weights_ptr, num_tokens, inv_double_sigma_sq, num_grid_points, mesh_cells_ptr
    )
    
    # 3. 물리 주소 딕셔너리 수입 (6대 필드 매핑)
    raw_views = wave_ingress_interface.bridge_raw_pointers_to_jax_view(mesh_cells_ptr, num_grid_points)
    
    # 4. 제로카피 승격 및 6대 필드 언팩 (부동소수점 + 정수형 제어 필드 포함)
    spatial_u = jnp.asarray(CUDAInterfaceBridge(raw_views["spatial_u"]))
    param_w = jnp.asarray(CUDAInterfaceBridge(raw_views["param_w"]))
    spatial_v = jnp.asarray(CUDAInterfaceBridge(raw_views["spatial_v"]))
    adaptive_gain = jnp.asarray(CUDAInterfaceBridge(raw_views["adaptive_gain"]))
    cell_status = jnp.asarray(CUDAInterfaceBridge(raw_views["cell_status"]))
    coordinate_id = jnp.asarray(CUDAInterfaceBridge(raw_views["coordinate_id"]))
    
    # 🛡 크리티컬 펜스: 비동기 실행 큐 보호
    for field in [spatial_u, param_w, spatial_v, adaptive_gain, cell_status, coordinate_id]:
        field.block_until_ready()
        
    return spatial_u, param_w, spatial_v, adaptive_gain, cell_status, coordinate_id

__all__ = ["fluidic_token_frontend_encoder", "CUDAInterfaceBridge"]
