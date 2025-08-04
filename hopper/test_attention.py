# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import traceback
from copy import deepcopy
import nvtx
try:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90
except:
    from sageattention import sageattn as sageattn_qk_int8_pv_fp8_cuda_sm90


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)

from spas_sage_attn import spas_sage2_attn_meansim_cuda

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    use_sage=True
):
    # q = q.transpose(1, 2).to(dtype)
    # k = k.transpose(1, 2).to(dtype)
    # v = v.transpose(1, 2).to(dtype)
    
    # out = spas_sage2_attn_meansim_cuda(q, k, v, simthreshd1=0.6, cdfthreshd=0.98, is_causal=False)
    # return out.transpose(1, 2).contiguous()

    if use_sage:
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = sageattn_qk_int8_pv_fp8_cuda_sm90(
            q, 
            k, 
            v,
            tensor_layout="HND",
            is_causal=causal,
            sm_scale=None,
            return_lse=False,
            pv_accum_dtype="fp32+fp32"
        )

        out = out.transpose(1, 2).contiguous()
        return out

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


shapes = [
    # ([1, 40, 729, 128], [1, 40, 512, 128], [1, 40, 512, 128]),
    # ([1, 40, 729, 128], [1, 40, 257, 128], [1, 40, 257, 128]),
    # ([1, 40, 729, 128], [1, 40, 729, 128], [1, 40, 729, 128]),
    # ([1, 40, 73920, 128], [1, 40, 257, 128], [1, 40, 257, 128]),
    # ([1, 40, 73920, 128], [1, 40, 73920, 128], [1, 40, 73920, 128]),
    # ([1, 16, 257, 80], [1, 16, 257, 80], [1, 16, 257, 80]),
    ([1, 40, 73920, 128], [1, 40, 512, 128], [1, 40, 512, 128])
    # ([1, 73920, 40, 128], [1, 512, 40, 128], [1, 512, 40, 128])
]
def run_attention(use_sage, fa_version):
    for shape in shapes:
        q_shape, k_shape, v_shape = deepcopy(shape)
        q_shape[1], q_shape[2] = q_shape[2], q_shape[1]
        k_shape[1], k_shape[2] = k_shape[2], k_shape[1]
        v_shape[1], v_shape[2] = v_shape[2], v_shape[1]

        q = torch.randn(q_shape, device='cuda', dtype=torch.float16)
        k = torch.randn(k_shape, device='cuda', dtype=torch.float16)
        v = torch.randn(v_shape, device='cuda', dtype=torch.float16)
        # if not use_sage:
        #     q.to(torch.float8_e4m3fn)
        #     k.to(torch.float8_e4m3fn)
        #     v.to(torch.float8_e4m3fn)
        try:
            # attention(q,k,v, use_sage=use_sage, dtype=torch.float8_e4m3fn if not use_sage else torch.float16, fa_version=fa_version)
            attention(q,k,v, use_sage=use_sage, dtype=torch.float16, fa_version=fa_version)
        except Exception as e:
            traceback.print_exc()
            print('Error: ',e)
            print(shape)
            print(use_sage)
            print(q_shape, k_shape, v_shape)

# run_attention(True, 3)
run_attention(False, 3)
