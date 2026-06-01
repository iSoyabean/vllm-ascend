#include "catccos/catccos_torch_adpt.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "catccos/matmul_allreduce/matmul_allreduce_wrapper.h"
#include "shmem.h"

namespace vllm_ascend {
namespace {

constexpr size_t kCatccosSymmetricWorkspaceBytes = 2ULL * 1024 * 1024 * 1024;
constexpr uint32_t kCatccosMmarBlockDim = 20;

struct CatccosState {
    bool initialized = false;
    void* symmetric_workspace = nullptr;
    size_t symmetric_workspace_bytes = 0;
    int64_t rank_id = -1;
    int64_t rank_size = 0;
};

int32_t set_catccos_init_attr(int32_t my_pe,
                              int32_t n_pes,
                              uint64_t local_mem_size,
                              const char* ip_port,
                              aclshmemx_init_attr_t* attributes,
                              aclshmemx_uniqueid_t* flag_uid)
{
    if (ip_port == nullptr) {
        return ACLSHMEM_INVALID_VALUE;
    }

    const size_t ip_len = std::min(std::strlen(ip_port), sizeof(attributes->ip_port) - 1);
    if (ip_len == 0) {
        return ACLSHMEM_INVALID_VALUE;
    }
    std::copy_n(ip_port, ip_len, attributes->ip_port);

    const int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    flag_uid->my_pe = my_pe;
    flag_uid->n_pes = n_pes;
    attributes->comm_args = reinterpret_cast<void*>(flag_uid);
    return ACLSHMEM_SUCCESS;
}

CatccosState& state()
{
    static CatccosState value;
    return value;
}

void* get_symmetric_workspace(size_t bytes)
{
    auto& runtime = state();
    TORCH_CHECK(runtime.initialized,
                "catccos is not initialized, call catccos_init first");

    if (runtime.symmetric_workspace == nullptr) {
        runtime.symmetric_workspace = shmem_malloc(bytes);
        TORCH_CHECK(runtime.symmetric_workspace != nullptr,
                    "catccos shmem_malloc failed, bytes=", bytes);
        runtime.symmetric_workspace_bytes = bytes;

        const auto ret = aclrtMemset(runtime.symmetric_workspace, bytes, 0, bytes);
        TORCH_CHECK(ret == ACL_SUCCESS,
                    "catccos aclrtMemset workspace failed, ret=", ret);
    } else {
        TORCH_CHECK(runtime.symmetric_workspace_bytes >= bytes,
                    "catccos symmetric workspace too small, existing=",
                    runtime.symmetric_workspace_bytes, ", required=", bytes);
    }

    return runtime.symmetric_workspace;
}

uint64_t get_ffts_config()
{
    TORCH_CHECK(state().initialized,
                "catccos is not initialized, call catccos_init first");
    const uint64_t ffts = shmemx_get_ffts_config();
    TORCH_CHECK(ffts != 0, "catccos shmemx_get_ffts_config returned 0");
    return ffts;
}

void check_mmar_inputs(const at::Tensor& a, const at::Tensor& b, int64_t rank_size)
{
    TORCH_CHECK(state().initialized,
                "catccos is not initialized, call catccos_init first");
    TORCH_CHECK(rank_size > 0, "catccos rank_size must be > 0, got ", rank_size);
    TORCH_CHECK(rank_size == state().rank_size,
                "catccos rank_size mismatch, input=", rank_size,
                ", initialized=", state().rank_size);

    TORCH_CHECK(a.dtype() == at::kHalf, "catccos MMAR supports float16 a only, got ", a.dtype());
    TORCH_CHECK(b.dtype() == at::kHalf, "catccos MMAR supports float16 b only, got ", b.dtype());
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
                "catccos MMAR requires NPU tensor a, got ", a.device());
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1,
                "catccos MMAR requires NPU tensor b, got ", b.device());
    TORCH_CHECK(a.device() == b.device(),
                "catccos MMAR requires a and b on same device, got a=",
                a.device(), ", b=", b.device());
    TORCH_CHECK(a.dim() == 2, "catccos MMAR requires 2D tensor a, got dim=", a.dim());
    TORCH_CHECK(b.dim() == 2, "catccos MMAR requires 2D tensor b, got dim=", b.dim());
    TORCH_CHECK(a.size(1) == b.size(0),
                "catccos MMAR shape mismatch, a.size(1)=", a.size(1),
                ", b.size(0)=", b.size(0));

    const int32_t n_pes = shmem_n_pes();
    TORCH_CHECK(n_pes > 0, "catccos shmem_n_pes invalid: ", n_pes);
    TORCH_CHECK(rank_size == n_pes,
                "catccos rank_size mismatch, input=", rank_size,
                ", shmem_n_pes=", n_pes);
}

}  // namespace

int64_t catccos_init(int64_t rank_id,
                     int64_t rank_size,
                     int64_t local_mem_size,
                     c10::string_view ip_port)
{
    auto& runtime = state();
    if (runtime.initialized) {
        return 0;
    }

    TORCH_CHECK(rank_id >= 0, "catccos rank_id must be >= 0, got ", rank_id);
    TORCH_CHECK(rank_size > 0, "catccos rank_size must be > 0, got ", rank_size);
    TORCH_CHECK(local_mem_size > 0,
                "catccos local_mem_size must be > 0, got ", local_mem_size);
    TORCH_CHECK(!ip_port.empty(), "catccos ip_port must not be empty");

    int64_t status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
    TORCH_CHECK(status == ACLSHMEM_SUCCESS,
                "catccos aclshmemx_set_conf_store_tls failed, status=", status);

    aclshmemx_uniqueid_t default_flag_uid{};
    aclshmemx_init_attr_t attributes{};
    const std::string ip_port_str(ip_port.data(), ip_port.size());
    status = set_catccos_init_attr(static_cast<int32_t>(rank_id),
                                   static_cast<int32_t>(rank_size),
                                   static_cast<uint64_t>(local_mem_size),
                                   ip_port_str.c_str(),
                                   &attributes,
                                   &default_flag_uid);
    TORCH_CHECK(status == ACLSHMEM_SUCCESS,
                "catccos set_attr failed, status=", status);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    TORCH_CHECK(status == ACLSHMEM_SUCCESS,
                "catccos aclshmemx_init_attr failed, status=", status);

    runtime.initialized = true;
    runtime.rank_id = rank_id;
    runtime.rank_size = rank_size;
    return 0;
}

at::Tensor catccos_matmul_allreduce(const at::Tensor& a,
                                    const at::Tensor& b,
                                    int64_t rank_size)
{
    c10_npu::OptionalNPUGuard guard(a.device());
    check_mmar_inputs(a, b, rank_size);

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    auto output = at::empty({m, n}, a.options()).contiguous();

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();

    auto* a_ptr = static_cast<uint8_t*>(a_contig.data_ptr());
    auto* b_ptr = static_cast<uint8_t*>(b_contig.data_ptr());
    auto* output_ptr = static_cast<uint8_t*>(output.data_ptr());
    auto* workspace_ptr = static_cast<uint8_t*>(
        get_symmetric_workspace(kCatccosSymmetricWorkspaceBytes));
    const uint64_t ffts = get_ffts_config();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    TORCH_CHECK(stream != nullptr, "catccos MMAR NPU stream is null");

    const int32_t rank_id = shmem_my_pe();
    const int32_t n_pes = shmem_n_pes();

    at_npu::native::OpCommand cmd;
    cmd.Name("catccos_matmul_allreduce");
    cmd.Input(a_contig);
    cmd.Input(b_contig);
    cmd.Output(output);
    cmd.SetCustomHandler([stream, ffts, a_ptr, b_ptr, output_ptr, workspace_ptr,
                          m, n, k, rank_id, n_pes]() -> int {
        CatccosKernel::catccos_matmul_allreduce_wrapper(
            kCatccosMmarBlockDim,
            stream,
            ffts,
            a_ptr,
            b_ptr,
            output_ptr,
            workspace_ptr,
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
            rank_id,
            n_pes);
        return 0;
    });
    cmd.Run();

    return output;
}

int64_t catccos_finalize()
{
    auto& runtime = state();
    if (!runtime.initialized) {
        return 0;
    }

    if (runtime.symmetric_workspace != nullptr) {
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        if (stream != nullptr) {
            const auto ret = aclrtSynchronizeStream(stream);
            TORCH_CHECK(ret == ACL_SUCCESS,
                        "catccos aclrtSynchronizeStream before free failed, ret=", ret);
        }
        shmem_free(runtime.symmetric_workspace);
        runtime.symmetric_workspace = nullptr;
        runtime.symmetric_workspace_bytes = 0;
    }

    const int64_t status = aclshmem_finalize();
    runtime.initialized = false;
    runtime.rank_id = -1;
    runtime.rank_size = 0;
    TORCH_CHECK(status == ACLSHMEM_SUCCESS,
                "catccos aclshmem_finalize failed, status=", status);
    return 0;
}

}  // namespace vllm_ascend
