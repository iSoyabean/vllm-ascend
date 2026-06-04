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

set(VLLM_ASCEND_CATCCOS_CCEC ${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng)
if(NOT EXISTS ${VLLM_ASCEND_CATCCOS_CCEC})
  set(VLLM_ASCEND_CATCCOS_CCEC ${ASCEND_HOME_PATH}/compiler/bin/bisheng)
endif()
if(NOT EXISTS ${VLLM_ASCEND_CATCCOS_CCEC})
  message(FATAL_ERROR "bisheng compiler not found for catccos build")
endif()

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

# Use the SHMEM copy carried by csrc/third_party/catccos. Do not consume
# SHMEM_HOME_PATH by default: system-level SHMEM may point to another version
# and make compile/link/runtime drift from the vendored catccos dependency.
set(VLLM_ASCEND_CATCCOS_SHMEM_ROOT
  ${VLLM_ASCEND_CATCCOS_ROOT}/3rdparty/shmem
)
set(VLLM_ASCEND_CATCCOS_SHMEM_INSTALL_DIR
  ${VLLM_ASCEND_CATCCOS_SHMEM_ROOT}/install/shmem
)
set(VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR
  ${VLLM_ASCEND_CATCCOS_SHMEM_INSTALL_DIR}/lib
)

if(NOT EXISTS ${VLLM_ASCEND_CATCCOS_SHMEM_ROOT}/CMakeLists.txt)
  message(FATAL_ERROR "catccos shmem source not found: ${VLLM_ASCEND_CATCCOS_SHMEM_ROOT}")
endif()

set(VLLM_ASCEND_CATCCOS_SHMEM_REQUIRED_LIBS
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}/libshmem.so
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}/libshmem_utils.so
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}/aclshmem_bootstrap_config_store.so
)

set(VLLM_ASCEND_CATCCOS_SHMEM_NEEDS_BUILD FALSE)
foreach(_CATCCOS_SHMEM_REQUIRED_LIB IN LISTS VLLM_ASCEND_CATCCOS_SHMEM_REQUIRED_LIBS)
  if(NOT EXISTS ${_CATCCOS_SHMEM_REQUIRED_LIB})
    set(VLLM_ASCEND_CATCCOS_SHMEM_NEEDS_BUILD TRUE)
  endif()
endforeach()
file(GLOB VLLM_ASCEND_CATCCOS_SHMEM_BOOTSTRAP_LIBS
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}/aclshmem_bootstrap_*.so*
)
if(NOT VLLM_ASCEND_CATCCOS_SHMEM_BOOTSTRAP_LIBS)
  set(VLLM_ASCEND_CATCCOS_SHMEM_NEEDS_BUILD TRUE)
endif()

if(VLLM_ASCEND_CATCCOS_SHMEM_NEEDS_BUILD)
  set(VLLM_ASCEND_CATCCOS_SHMEM_BUILD_DIR
    ${CMAKE_BINARY_DIR}/catccos_shmem_build
  )
  set(VLLM_ASCEND_CATCCOS_SHMEM_BUILD_STAMP
    ${CMAKE_CURRENT_BINARY_DIR}/catccos_shmem_build.stamp
  )
  add_custom_command(
    OUTPUT ${VLLM_ASCEND_CATCCOS_SHMEM_BUILD_STAMP}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VLLM_ASCEND_CATCCOS_SHMEM_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E env
            ASCEND_HOME_PATH=${ASCEND_HOME_PATH}
            ASCEND_TOOLKIT_HOME=${ASCEND_HOME_PATH}
            ${CMAKE_COMMAND}
            -S ${VLLM_ASCEND_CATCCOS_SHMEM_ROOT}
            -B ${VLLM_ASCEND_CATCCOS_SHMEM_BUILD_DIR}
            -DCMAKE_INSTALL_PREFIX=${VLLM_ASCEND_CATCCOS_SHMEM_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=RELEASE
            -DCMAKE_C_COMPILER=${VLLM_ASCEND_CATCCOS_CCEC}
            -DCMAKE_CXX_COMPILER=${VLLM_ASCEND_CATCCOS_CCEC}
            -DUSE_CXX11_ABI=ON
            -DUSE_MSSANITIZER=OFF
            -DSOC_TYPE=${SOC_VERSION}
            -DPYEXPAND_EXAMPLE=OFF
    COMMAND ${CMAKE_COMMAND} --build ${VLLM_ASCEND_CATCCOS_SHMEM_BUILD_DIR}
            --target install --parallel
    COMMAND ${CMAKE_COMMAND} -E touch ${VLLM_ASCEND_CATCCOS_SHMEM_BUILD_STAMP}
    BYPRODUCTS ${VLLM_ASCEND_CATCCOS_SHMEM_REQUIRED_LIBS}
    VERBATIM
    COMMENT "Building bundled catccos SHMEM"
  )
  add_custom_target(catccos_shmem_build
    DEPENDS ${VLLM_ASCEND_CATCCOS_SHMEM_BUILD_STAMP}
  )
else()
  message(STATUS "catccos bundled SHMEM detected: ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}")
  add_custom_target(catccos_shmem_build)
endif()

set(VLLM_ASCEND_CATCCOS_SHMEM_LINK_LIBS
  shmem
  shmem_utils
)

set(VLLM_ASCEND_CATCCOS_SHMEM_INSTALL_CODE
  "
  file(GLOB _CATCCOS_SHMEM_RUNTIME_LIBS
    \"${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}/*.so*\"
  )
  if(NOT _CATCCOS_SHMEM_RUNTIME_LIBS)
    message(FATAL_ERROR \"catccos shmem runtime libraries not found in ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}\")
  endif()
  file(INSTALL DESTINATION \"${VLLM_ASCEND_INSTALL_PATH}/lib\"
       TYPE FILE FILES \${_CATCCOS_SHMEM_RUNTIME_LIBS})
  "
)

# Device wrapper: compile the catccos MMAR launch wrapper with CCEC/AscendC,
# following catccos examples instead of appending it to a host C++ target.
set(VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO
  ${CMAKE_BINARY_DIR}/lib/libcatccos_mmar_kernel.so
)

set(VLLM_ASCEND_CATCCOS_KERNEL_INCLUDE_FLAGS)
foreach(_CATCCOS_KERNEL_INCLUDE_DIR IN LISTS VLLM_ASCEND_CATCCOS_INCLUDE_DIRS)
  list(APPEND VLLM_ASCEND_CATCCOS_KERNEL_INCLUDE_FLAGS -I${_CATCCOS_KERNEL_INCLUDE_DIR})
endforeach()
list(APPEND VLLM_ASCEND_CATCCOS_KERNEL_INCLUDE_FLAGS
  -I${ASCEND_HOME_PATH}/include
)

set(VLLM_ASCEND_CATCCOS_KERNEL_LINK_FLAGS
  -L${ASCEND_HOME_PATH}/lib64
  -L${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}
)

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
add_dependencies(catccos_mmar_kernel_build catccos_shmem_build)

# Host adapter: torch/NPU argument checks, SHMEM lifecycle, and OpCommand
# custom handler entry live in vllm_ascend_C.
target_sources(vllm_ascend_C PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/catccos_torch_adpt.cpp
)
target_include_directories(vllm_ascend_C PRIVATE
  ${VLLM_ASCEND_CATCCOS_INCLUDE_DIRS}
)
target_link_directories(vllm_ascend_C PRIVATE
  ${VLLM_ASCEND_CATCCOS_SHMEM_LIB_DIR}
)
target_link_libraries(vllm_ascend_C PUBLIC
  ${VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO}
  ${VLLM_ASCEND_CATCCOS_SHMEM_LINK_LIBS}
)
add_dependencies(vllm_ascend_C catccos_mmar_kernel_build catccos_shmem_build)

install(FILES ${VLLM_ASCEND_CATCCOS_MMAR_KERNEL_SO}
  DESTINATION ${VLLM_ASCEND_INSTALL_PATH}
)
install(CODE "${VLLM_ASCEND_CATCCOS_SHMEM_INSTALL_CODE}")
