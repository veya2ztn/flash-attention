#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
inline __device__ void apply_alibi(Tensor<Engine, Layout> &tensor, 
                                   const int col_idx_offset_,
                                   const int max_seqlen_k, 
                                   const int row_idx_offset_,
                                   const int max_seqlen_q, 
                                   const int warp_row_stride,
                                   const int head_idx,
                                   const float softmax_scale,
                                   const float alibi_start,
                                   const float alibi_ratio) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int row_idx_offset = row_idx_offset_;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    float alibi_slope;
    if (alibi_ratio == -1) {
        float fixed_values[] = {-0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125, -0.00390625, -0.70710677, -0.35355338, -0.17677668, -0.08838834};
        alibi_slope = fixed_values[head_idx % 12];
    } else if (alibi_ratio == -2) {
        float fixed_values[] = {-0.5, -0.23473272, -0.11019891, -0.05173458, -0.0242876, -0.01140219, -0.00535293, -0.00251302, -0.70710677, -0.35355338, -0.17677668, -0.08838834};
        alibi_slope = fixed_values[head_idx % 12];
    } else if (alibi_ratio < 0) {
        // When alibi_ratio is less than 0 but not -1 or -2, throw an error
        // Note: CUDA does not support exceptions so we print an error message instead
        printf("Error: Invalid alibi_ratio value. It must be either -1 or -2 when it's less than 0.\n");
        return;
    } else {
        alibi_slope = alibi_start * powf(alibi_ratio, (float)head_idx);
    }
    int offset;
    float alibi;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (col_idx < row_idx)
                    {
                        offset =  (row_idx - col_idx) ;
                    }
                    else{
                        offset =  (col_idx - row_idx) ;
                    }
                    alibi = (alibi_slope * offset) / softmax_scale;
                    // const float alibi = (alibi_slope * col_idx) / softmax_scale;
                    if (col_idx < max_seqlen_k && row_idx < max_seqlen_q) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi;
                    }
                }
            }
        }
    }
}

}  // namespace flash