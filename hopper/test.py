# import cudnn
# import torch
# import math

# torch.manual_seed(42)
# handle = cudnn.create_handle()

# assert torch.cuda.is_available()
# assert (
#     torch.cuda.get_device_capability()[0] >= 8
# ), "SDPA operation is only supported on SM80 architecture (Ampere) or above"
# assert (
#     cudnn.backend_version() >= 8903
# ), "SDPA operation is only supported cuDNN version 8.9.3 or above"
# b = 1  # batch size
# h = 40  # query number of heads
# s = 73920  # maximum sequence length
# d = 128  # embedding dimension per head
# s_kv = 512
# # ([1, 40, 73920, 128], [1, 40, 512, 128], [1, 40, 512, 128])


# attn_scale = 1.0 / math.sqrt(d)

# # The tensors will have non-interleaved
# # BHSD (batch, num_head, sequence_length, dims_per_head) logical tensor layout
# dims = (b, h, s, d)
# # BSHD (batch, sequence_length, num_head, dims_per_head) physical layout
# strides = (s * h * d, d, h * d, 1)
# # For BHSD (batch, num_head, sequence_length, dims_per_head) physical tensor layout, uncomment the following:
# # strides = (s * h * d, s * d, d, 1)

# q_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
# k_gpu = torch.randn(b * s_kv * h * d).half().cuda().as_strided((b, h, s_kv, d), (s_kv * h * d, d, h * d, 1))
# v_gpu = torch.randn(b * s_kv * h * d).half().cuda().as_strided((b, h, s_kv, d), (s_kv * h * d, d, h * d, 1))
# o_gpu = torch.empty(b * s * h * d).half().cuda().as_strided(dims, strides)

# graph = cudnn.pygraph(
#     io_data_type=cudnn.data_type.HALF,
#     intermediate_data_type=cudnn.data_type.FLOAT,
#     compute_data_type=cudnn.data_type.FLOAT,
# )

# q = graph.tensor_like(q_gpu)
# k = graph.tensor_like(k_gpu)
# v = graph.tensor_like(v_gpu)

# # the second return for the stats tensor is used for training only.
# # causal mask is enabled
# o, _ = graph.sdpa(
#     name="sdpa",
#     q=q,
#     k=k,
#     v=v,
#     generate_stats=False,
#     attn_scale=attn_scale,
#     use_causal_mask=True,
# )

# o.set_output(True).set_dim(dims).set_stride(strides)

# graph.validate()
# graph.build_operation_graph()
# graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
# graph.check_support()
# graph.build_plans()

# variant_pack = {
#     q: q_gpu,
#     k: k_gpu,
#     v: v_gpu,
#     o: o_gpu,
# }

# workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
# graph.execute(variant_pack, workspace)
# torch.cuda.synchronize()

# # q_ref = q_gpu.detach().float().requires_grad_()
# # k_ref = k_gpu.detach().float().requires_grad_()
# # v_ref = v_gpu.detach().float().requires_grad_()

# # o_ref = torch.nn.functional.scaled_dot_product_attention(
# #     q_ref, k_ref, v_ref, is_causal=True, scale=attn_scale
# # )
# # torch.testing.assert_close(o_ref, o_gpu.float(), atol=5e-3, rtol=3e-3)


# from flash_attn_3._C import 