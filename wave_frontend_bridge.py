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
        self.__cuda_array_interface__: Dict[str, Any] = interface_dict.get("__cuda_array_interface__", interface_dict)
        
        # [🛡 SILICON VALIDITY GUARD] JAX/XLA가 하이재킹할 필수 3대 원소("data", "shape", "typestr") 유효성 최종 확인
        # 포인터 주소(data), 형상(shape), 데이터 타입(typestr) 중 하나라도 누락되면 XLA 컴파일 시 단독 크래시가 유발됩니다.
        assert all(key in self.__cuda_array_interface__ for key in ("data", "shape", "typestr")), (
            "[🚨 CUDAInterfaceBridge] Invalid CUDA Array Interface layout. "
            "The hardware-level pointer constraints ('data', 'shape', 'typestr') keys must be defined."
        )


    # 3. [⛓ 0ns CONNECTOR ENGAGEMENT] C++ 구조체 배열에서 offsetof 규격으로 가공된 물리 주소선 딕셔너리 수입
    # 이 딕셔너리는 메모리 복사본이 아닌, VRAM 물리 주소 명세서 그 자체입니다.
    raw_views: dict = wave_ingress_interface.bridge_raw_pointers_to_jax_view(mesh_cells_ptr, num_grid_points)
    
    # 4. [👑 ZERO-COPY ASCENSION] 가속기 전역 메모리 복사 없이 JAX 고차원 텐서로 즉시 승격
    # 3대 핵심 원소("data", "shape", "typestr") 검증 가드가 심어진 CUDAInterfaceBridge를 거치며
    # 32바이트 보폭(Stride) 명세가 살아있는 채로 JAX/XLA 컴파일러 초입에 하이재킹이 안전하게 집행됩니다.
    spatial_u_field: jax.Array = jnp.asarray(CUDAInterfaceBridge(raw_views["spatial_u"]))
    param_w_field: jax.Array = jnp.asarray(CUDAInterfaceBridge(raw_views["param_w"]))
    spatial_v_field: jax.Array = jnp.asarray(CUDAInterfaceBridge(raw_views["spatial_v"]))
    adaptive_gain_field: jax.Array = jnp.asarray(CUDAInterfaceBridge(raw_views["adaptive_gain"]))
    
    # [🛡 CRITICAL LIFE-CYCLE SAFETY FENCE]
    # JAX의 비동기 실행 큐(Asynchronous Dispatch Queue)가 전사 연산을 완전히 접수할 때까지 
    # 하부 C++ 포인터 메모리 블록이 파손되거나 GC 타이밍 어긋남으로 증발하는 것을 방어합니다.
    # 이 연산선 하드 레이턴시 펜스 덕분에 CUDA Illegal Memory Access 크래시를 원천 차단합니다.
    spatial_u_field.block_until_ready()
    param_w_field.block_until_ready()
    spatial_v_field.block_until_ready()
    adaptive_gain_field.block_until_ready()
    
    return spatial_u_field, param_w_field, spatial_v_field, adaptive_gain_field


# 상위 거버넌스(main_orchestrator.py) 단으로 방출할 정적 안전 익스포트 목록 봉인
__all__ = ["fluidic_token_frontend_encoder", "CUDAInterfaceBridge"]
