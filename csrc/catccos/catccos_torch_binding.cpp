#include <torch/extension.h>
#include <torch/library.h>

#include "catccos/catccos_torch_adpt.h"
#include "utils.h"

REGISTER_EXTENSION(vllm_ascend_catccos_C)

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("catccos_init(int rank_id, int rank_size, int local_mem_size, str ip_port) -> int");
    ops.def("catccos_matmul_allreduce(Tensor a, Tensor b, int rank_size) -> Tensor");
    ops.def("catccos_finalize() -> int");

    ops.impl("catccos_init", &vllm_ascend::catccos_init);
    ops.impl("catccos_matmul_allreduce", torch::kPrivateUse1, &vllm_ascend::catccos_matmul_allreduce);
    ops.impl("catccos_finalize", &vllm_ascend::catccos_finalize);
}
