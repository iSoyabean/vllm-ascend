#include <torch/extension.h>
#include <torch/library.h>
#include <vector>

namespace vllm_ascend {
namespace meta {

at::Tensor catccos_matmul_allreduce_meta(const at::Tensor& a,
                                         const at::Tensor& b,
                                         int64_t rank_size)
{
    (void)rank_size;
    auto a_sizes = a.sym_sizes();
    auto b_sizes = b.sym_sizes();
    std::vector<c10::SymInt> out_shape{a_sizes[0], b_sizes[1]};
    return at::empty_symint(out_shape, a.options());
}

}  // namespace meta
}  // namespace vllm_ascend

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)
{
    ops.impl("catccos_matmul_allreduce", &vllm_ascend::meta::catccos_matmul_allreduce_meta);
}
