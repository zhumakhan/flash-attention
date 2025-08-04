#include <cstdio>
#include <cuda_runtime_api.h>
#include "flash_api.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

int main(){
    int batch_size = 1;
    int seqlen_q = 73920;
    int seqlen_k = 256;
    int heads = 40;
    int dim = 128;
    using half_t = __half;

    half_t *q;
    half_t *k;
    half_t *v;
    half_t *out;
    half_t *softmax_lse_ptr;

    cudaMalloc(reinterpret_cast<void**>(&q), batch_size * seqlen_q * heads * dim * sizeof(half_t));
    cudaMalloc(reinterpret_cast<void**>(&k), batch_size * seqlen_k * heads * dim * sizeof(half_t));
    cudaMalloc(reinterpret_cast<void**>(&v), batch_size * seqlen_k * heads * dim * sizeof(half_t));
    cudaMalloc(reinterpret_cast<void**>(&out), batch_size * seqlen_q * heads * dim * sizeof(half_t));
    cudaMalloc(reinterpret_cast<void**>(&softmax_lse_ptr), batch_size * seqlen_q * heads * dim * sizeof(half_t));


    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "Error creating CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    mha_fwd(q,k,v,out,softmax_lse_ptr, batch_size, seqlen_q, seqlen_k, heads, dim, stream);
    printf("\nHere we go!\n");
}