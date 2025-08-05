#pragma once

#include "flash.h"

int mha_fwd(void * q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
            void * k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
            void * v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
            void * out,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
            void * softmax_lse,
            const int batch_size,
            const int seqlen_q,
            const int seqlen_k,
            const int heads,
            const int dim,
            bool is_bf16,
            cudaStream_t stream);