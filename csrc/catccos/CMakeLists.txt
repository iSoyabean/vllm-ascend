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

# catccos/catlass device headers include AscendC headers such as
# kernel_operator.h. Keep this local to catccos targets and append only paths
# that exist in the active CANN installation.
set(VLLM_ASCEND_CATCCOS_ASCENDC_INCLUDE_DIRS)
foreach(_CATCCOS_ASCENDC_INCLUDE_DIR IN ITEMS
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include
    ${ASCEND_HOME_PATH}/arm64-linux/asc/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface)
  if(EXISTS ${_CATCCOS_ASCENDC_INCLUDE_DIR})
    list(APPEND VLLM_ASCEND_CATCCOS_ASCENDC_INCLUDE_DIRS ${_CATCCOS_ASCENDC_INCLUDE_DIR})
  endif()
endforeach()

# Use vllm-ascend's catlass provider, while catccos contributes only its
# headers and shmem dependency. Do not add catccos/3rdparty/catlass here.
set(VLLM_ASCEND_CATCCOS_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catlass/include
  ${VLLM_ASCEND_CATCCOS_ROOT}/include
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/shmem/include
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/shmem/src/device
  ${VLLM_ASCEND_CATCCOS_ASCENDC_INCLUDE_DIRS}
)

set(VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIRS
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/shmem/install/shmem/lib
)
if(DEFINED ENV{SHMEM_HOME_PATH})
  list(APPEND VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIRS $ENV{SHMEM_HOME_PATH}/shmem/lib)
endif()

set(VLLM_ASCEND_CATCCOS_SHMEM_LIBS shmem shmem_utils)

# Device wrapper: compile the catccos MMAR launch wrapper with CCEC/AscendC,
# following catccos examples instead of appending it to a host C++ target.
set(VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO
  ${CMAKE_BINARY_DIR}/lib/libcatccos_mmar_kernel.so
)

set(VLLM_ASCEND_CATCCOS_CCEC ${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng)
if(NOT EXISTS ${VLLM_ASCEND_CATCCOS_CCEC})
  set(VLLM_ASCEND_CATCCOS_CCEC ${ASCEND_HOME_PATH}/compiler/bin/bisheng)
endif()
if(NOT EXISTS ${VLLM_ASCEND_CATCCOS_CCEC})
  message(FATAL_ERROR "bisheng compiler not found for catccos MMAR kernel build")
endif()

set(VLLM_ASCEND_CATCCOS_KERNEL_INCLUDE_FLAGS)
foreach(_CATCCOS_KERNEL_INCLUDE_DIR IN LISTS VLLM_ASCEND_CATCCOS_INCLUDE_DIRS)
  list(APPEND VLLM_ASCEND_CATCCOS_KERNEL_INCLUDE_FLAGS -I${_CATCCOS_KERNEL_INCLUDE_DIR})
endforeach()
list(APPEND VLLM_ASCEND_CATCCOS_KERNEL_INCLUDE_FLAGS
  -I${ASCEND_HOME_PATH}/include
)

set(VLLM_ASCEND_CATCCOS_KERNEL_LINK_FLAGS
  -L${ASCEND_HOME_PATH}/lib64
)
foreach(_CATCCOS_SHMEM_LIB_DIR IN LISTS VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIRS)
  list(APPEND VLLM_ASCEND_CATCCOS_KERNEL_LINK_FLAGS -L${_CATCCOS_SHMEM_LIB_DIR})
endforeach()

add_custom_command(
  OUTPUT ${VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
  COMMAND ${VLLM_ASCEND_CATCCOS_CCEC}
          -O2 -shared -fPIC -std=c++17 -xcce --cce-aicore-arch=dav-c220
          -mllvm -cce-aicore-stack-size=0x8000
          -mllvm -cce-aicore-function-stack-size=0x8000
          -mllvm -cce-aicore-record-overflow=true
          -mllvm -cce-aicore-addr-transform
          -mllvm -cce-aicore-dcci-insert-for-scalar=false
          -DL2_CACHE_HINT
          -DCATLASS_ARCH=2201
          -MD -MF ${CMAKE_CURRENT_BINARY_DIR}/catccos_mmar_kernel.d
          ${VLLM_ASCEND_CATCCOS_KERNEL_INCLUDE_FLAGS}
          ${VLLM_ASCEND_CATCCOS_KERNEL_LINK_FLAGS}
          -lruntime -lstdc++ -lm -lascendcl
          -lshmem -lgraph_base -ltiling_api -lplatform -lregister
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul_allreduce/matmul_allreduce_wrapper.cpp
          -o ${VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_allreduce/matmul_allreduce_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_allreduce/matmul_allreduce_device.h
  DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/catccos_mmar_kernel.d
  VERBATIM
  COMMENT "Generating catccos MMAR kernel library"
)

add_custom_target(catccos_mmar_kernel_build
  DEPENDS ${VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO}
)

# Host adapter: torch/NPU argument checks, SHMEM lifecycle, and OpCommand
# custom handler entry live in vllm_ascend_C.
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
  ${VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO}
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIBS}
)
add_dependencies(vllm_ascend_C catccos_mmar_kernel_build)

install(FILES ${VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO}
  DESTINATION ${VLLM_ASCEND_INSTALL_PATH}
)
