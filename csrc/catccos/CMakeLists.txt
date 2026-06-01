# Local build wiring for catccos deep integration.
# The root CMake only adds this subdirectory; catccos-specific includes,
# sources, compile definitions, and shmem links stay scoped here.

set(VLLM_ASCEND_CATCCOS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catccos)

if(NOT EXISTS ${VLLM_ASCEND_CATCCOS_ROOT})
  message(STATUS "catccos source not detected, skip catccos deep integration build")
  return()
endif()

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
  message(STATUS "catccos source detected but skipped on this SOC: ${VLLM_ASCEND_CATCCOS_ROOT}")
  return()
endif()

message(STATUS "catccos source detected: ${VLLM_ASCEND_CATCCOS_ROOT}")

set(VLLM_ASCEND_CATCCOS_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${VLLM_ASCEND_CATCCOS_ROOT}/include
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/shmem/include
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/shmem/src/device
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/catlass/include
)

set(VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIRS
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/shmem/install/shmem/lib
)
if(DEFINED ENV{SHMEM_HOME_PATH})
  list(APPEND VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIRS $ENV{SHMEM_HOME_PATH}/shmem/lib)
endif()

set(VLLM_ASCEND_CATCCOS_SHMEM_LIBS shmem shmem_utils)

target_sources(vllm_ascend_C PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/catccos_torch_adpt.cpp
)
target_include_directories(vllm_ascend_C PRIVATE
  ${VLLM_ASCEND_CATCCOS_INCLUDE_DIRS}
)
target_link_directories(vllm_ascend_C PRIVATE
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIRS}
)
target_link_libraries(vllm_ascend_C PUBLIC
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIBS}
)

if(TARGET vllm_ascend_kernels)
  target_sources(vllm_ascend_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_allreduce/matmul_allreduce_wrapper.cpp
  )
  target_include_directories(vllm_ascend_kernels PRIVATE
    ${VLLM_ASCEND_CATCCOS_INCLUDE_DIRS}
  )
  target_link_directories(vllm_ascend_kernels PRIVATE
    ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIRS}
  )
  target_link_libraries(vllm_ascend_kernels PRIVATE
    ${VLLM_ASCEND_CATCCOS_SHMEM_LIBS}
  )
  target_compile_definitions(vllm_ascend_kernels PRIVATE CATLASS_ARCH=2201)
endif()
