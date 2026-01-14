"""Resource estimates for tests, generated internally by utils/generate_test_resources_report"""
TEST_RESOURCES = {
    "//max/examples/capi:test": {
        "cpu": 2,
        "memory": 1219,
    },
    "//max/examples/custom-graph-module:main_test": {
        "cpu": 2,
    },
    "//max/examples/custom_ops:addition.example-test": {
        "cpu": 2,
        "memory": 1605,
    },
    "//max/examples/custom_ops:histogram.example-test": {
        "cpu": 2,
        "memory": 1418,
    },
    "//max/examples/custom_ops:image_pipeline.example-test": {
        "cpu": 3,
        "memory": 1226,
    },
    "//max/examples/custom_ops:mandelbrot.example-test": {
        "cpu": 2,
        "memory": 1202,
    },
    "//max/examples/custom_ops:parametric_addition.example-test": {
        "cpu": 2,
        "memory": 1395,
    },
    "//max/examples/custom_ops:top_k.example-test": {
        "cpu": 2,
        "memory": 1890,
    },
    "//max/examples/custom_ops:vector_addition.example-test": {
        "cpu": 2,
        "memory": 1363,
    },
    "//max/examples/max-graph:addition_test": {
        "cpu": 3,
        "memory": 1146,
    },
    "//max/examples/offline-inference:basic_test": {
        "cpu": 4,
        "memory": 44163,
    },
    "//max/examples/pytorch_custom_ops:addition.example-test": {
        "memory": 1730,
    },
    "//max/examples/pytorch_custom_ops:graph.example-test": {
        "cpu": 2,
        "memory": 2953,
    },
    "//max/examples/pytorch_custom_ops:grayscale.example-test": {
        "cpu": 2,
        "memory": 1936,
    },
    "//max/kernels/benchmarks/autotune:tests/test_kbench": {
        "cpu": 4,
        "memory": 2061,
    },
    "//max/kernels/benchmarks:algorithm/parallelize_overhead.mojo.test": {
        "memory": 800,
    },
    "//max/kernels/test/gpu/linalg:test_matmul_sm100_ptx.mojo.test": {
        "memory": 300,
    },
    "//max/kernels/test/kv_cache:test_mha_mixed_ce_tg.mojo.test": {
        "memory": 194,
    },
    "//max/kernels/test/linalg:test_gemv.mojo.test": {
        "cpu": 8,
        "memory": 194,
    },
    "//max/kernels/test/linalg:test_neon_dotprod_intrinsics.mojo.test": {
        "memory": 300,
    },
    "//max/kernels/test/linalg:test_neon_matmul_intrinsics.mojo.test": {
        "memory": 300,
    },
    "//max/kernels/test/linalg:test_vnni_intrinsics.mojo.test": {
        "cpu": 3,
        "memory": 419,
    },
    "//max/kernels/test/nn:test_conv1d.mojo.test": {
        "memory": 109,
    },
    "//max/kernels/test/nn:test_direct_conv.mojo.test": {
        "memory": 584,
    },
    "//max/kernels/test/nn:test_rms_norm_fused_residual_add.mojo.test": {
        "cpu": 2,
    },
    "//max/kernels/test/nn:test_top_k.mojo.test": {
        "cpu": 2,
    },
    "//max/kernels/test/nn:test_toppminp.mojo.test": {
        "memory": 196,
    },
    "//max/kernels/test/quantization:test_qmatmul_k.mojo.test": {
        "cpu": 6,
    },
    "//max/tests/integration/API/python/graph:test_matmul_packed": {
        "cpu": 2,
        "memory": 5340,
    },
    "//max/tests/integration/API/python/graph:test_reduce_add": {
        "cpu": 2,
        "memory": 1474,
    },
    "//max/tests/integration/API/python/interfaces:test_hash_image": {
        "cpu": 3,
        "memory": 300,
    },
    "//max/tests/integration/API/python/interfaces:test_queue": {
        "cpu": 5,
        "memory": 300,
    },
    "//max/tests/integration/API/python/interfaces:test_serialization": {
        "cpu": 4,
        "memory": 300,
    },
    "//max/tests/integration/API/python/interfaces:test_tokens": {
        "cpu": 4,
        "memory": 300,
    },
    "//max/tests/integration/API/python/interfaces:text_generation/test_text_generation_request": {
        "cpu": 3,
        "memory": 300,
    },
    "//max/tests/integration/API/python/nn/module_v3:norm/test_rms_norm": {
        "cpu": 8,
        "memory": 18892,
    },
    "//max/tests/integration/API/python/nn/module_v3:rope/test_rope": {
        "cpu": 8,
        "memory": 27579,
    },
    "//max/tests/integration/API/python/nn/module_v3:rope/test_yarn": {
        "cpu": 3,
        "memory": 11296,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_embedding": {
        "cpu": 5,
        "memory": 14107,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_linear": {
        "cpu": 8,
        "memory": 13476,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_module": {
        "cpu": 8,
        "memory": 38601,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_module_gpt2": {
        "cpu": 4,
        "memory": 17537,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_sequential": {
        "cpu": 6,
        "memory": 4860,
    },
    "//max/tests/integration/API/python/tensor:test_arange": {
        "cpu": 8,
        "memory": 39007,
    },
    "//max/tests/integration/API/python/tensor:test_functional_binary": {
        "cpu": 8,
        "memory": 39960,
    },
    "//max/tests/integration/API/python/tensor:test_functional_custom": {
        "cpu": 8,
        "memory": 3784,
    },
    "//max/tests/integration/API/python/tensor:test_functional_other": {
        "cpu": 8,
        "memory": 97997,
    },
    "//max/tests/integration/API/python/tensor:test_functional_reduction": {
        "cpu": 8,
        "memory": 12053,
    },
    "//max/tests/integration/API/python/tensor:test_functional_unary": {
        "cpu": 8,
        "memory": 54026,
    },
    "//max/tests/integration/API/python/tensor:test_random": {
        "cpu": 8,
        "memory": 26757,
    },
    "//max/tests/integration/API/python/tensor:test_tensor_elemwise": {
        "cpu": 8,
        "memory": 96279,
    },
    "//max/tests/integration/API/python/tensor:test_tensor_matmul": {
        "cpu": 8,
        "memory": 11224,
    },
    "//max/tests/integration/API/python/tensor:test_tensor_repr": {
        "cpu": 8,
        "memory": 36788,
    },
    "//max/tests/integration/API/python:test_load_library": {
        "cpu": 5,
    },
    "//max/tests/integration/API/python:test_load_library_3.10": {
        "cpu": 7,
    },
    "//max/tests/integration/API/python:test_load_library_3.11": {
        "cpu": 6,
    },
    "//max/tests/integration/API/python:test_load_library_3.13": {
        "cpu": 7,
    },
    "//max/tests/integration/API/python:test_load_library_3.14": {
        "cpu": 5,
    },
    "//max/tests/integration/API/python:tests-fail-weight-loading": {
        "cpu": 5,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_causal_attention_mask": {
        "memory": 300,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_causal_attention_mask_with_alibi": {
        "memory": 411,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_collate_batch": {
        "cpu": 3,
        "memory": 300,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_max_tokens_to_generate": {
        "cpu": 5,
        "memory": 300,
    },
    "//max/tests/integration/pipelines/python/kv_cache/attention:attention_no_opaque_tests": {
        "cpu": 2,
        "memory": 3705,
    },
    "//max/tests/integration/pipelines/python/kv_cache/attention:attention_tests": {
        "cpu": 2,
        "memory": 5623,
    },
    "//max/tests/integration/pipelines/python/kv_cache/transfer_engine:test_notification_latency": {
        "cpu": 2,
        "memory": 1740,
    },
    "//max/tests/integration/pipelines/python/kv_cache/transfer_engine:test_send_recv": {
        "cpu": 3,
        "memory": 685,
    },
    "//max/tests/integration/pipelines/python/kv_cache:embedding": {
        "cpu": 8,
        "memory": 21692,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_kv_cache_matmul": {
        "cpu": 2,
        "memory": 5784,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_memory_estimation": {
        "cpu": 2,
        "memory": 246,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_prefix_caching": {
        "cpu": 2,
        "memory": 2054,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_print_kv_cache": {
        "cpu": 2,
        "memory": 10948,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_rms_norm_key_cache": {
        "cpu": 2,
        "memory": 4307,
    },
    "//max/tests/integration/pipelines/python/mistral3:tests": {
        "cpu": 2,
        "memory": 672,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_block_hasher": {
        "cpu": 2,
        "memory": 819,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_cache_params": {
        "cpu": 2,
        "memory": 247,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_data_parallelism_utils": {
        "memory": 247,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_kv_cache_manager": {
        "cpu": 2,
        "memory": 2229,
    },
    "//max/tests/integration/pipelines/python/nn/norm:norm_tests": {
        "cpu": 2,
        "memory": 2999,
    },
    "//max/tests/integration/pipelines/python/nn:test_conv": {
        "cpu": 8,
        "memory": 17630,
    },
    "//max/tests/integration/pipelines/python/nn:test_identity": {
        "cpu": 6,
        "memory": 6774,
    },
    "//max/tests/integration/pipelines/python/nn:test_layer_hook": {
        "cpu": 8,
        "memory": 7299,
    },
    "//max/tests/integration/pipelines/python/nn:test_mlp": {
        "cpu": 8,
        "memory": 55325,
    },
    "//max/tests/integration/pipelines/python/nn:test_print_hook": {
        "cpu": 8,
        "memory": 8380,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_compute_log_probabilities": {
        "cpu": 2,
        "memory": 1876,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_lora_graph_inputs": {
        "cpu": 2,
        "memory": 854,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_pipeline_lora_sorting": {
        "cpu": 2,
        "memory": 808,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_text_generation_pipeline": {
        "cpu": 2,
        "memory": 2161,
    },
    "//max/tests/integration/pipelines/python/qwen2_5vl:test_compute_scatter_gather_indices": {
        "cpu": 2,
        "memory": 654,
    },
    "//max/tests/integration/pipelines/python/qwen2_5vl:test_vision_functions": {
        "memory": 12796,
    },
    "//max/tests/integration/pipelines/python/qwen3vl:test_vision_functions": {
        "cpu": 2,
        "memory": 817,
    },
    "//max/tests/integration/pipelines/python/whisper:whisper": {
        "cpu": 2,
        "memory": 827,
    },
    "//max/tests/integration/pipelines/python:test_debug_model": {
        "cpu": 2,
        "memory": 816,
    },
    "//max/tests/integration/pipelines/python:test_debug_utils": {
        "memory": 828,
    },
    "//max/tests/integration/pipelines/python:test_hf_config_overrides": {
        "cpu": 2,
        "memory": 676,
    },
    "//max/tests/integration/pipelines/python:test_hf_repo_lock": {
        "memory": 856,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli": {
        "cpu": 3,
        "memory": 6413,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli_help": {
        "cpu": 2,
        "memory": 684,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli_json_lightweight": {
        "cpu": 2,
        "memory": 669,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli_lightweight": {
        "cpu": 2,
        "memory": 710,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_lm_eval": {
        "cpu": 2,
        "memory": 8006,
    },
    "//max/tests/integration/serve/kvcache_agent:tests": {
        "cpu": 3,
        "memory": 105,
    },
    "//max/tests/integration/serve:test_sagemaker_cpu": {
        "memory": 1053,
    },
    "//max/tests/integration/serve:test_stop_cpu": {
        "memory": 1091,
    },
    "//max/tests/tests/_core_mojo:tests": {
        "cpu": 3,
        "memory": 616,
    },
    "//max/tests/tests/driver:test_device": {
        "cpu": 2,
        "memory": 221,
    },
    "//max/tests/tests/driver:test_driver": {
        "cpu": 2,
        "memory": 252,
    },
    "//max/tests/tests/driver:test_tensor": {
        "cpu": 2,
        "memory": 223,
    },
    "//max/tests/tests/entrypoints:tests": {
        "cpu": 2,
        "memory": 667,
    },
    "//max/tests/tests/graph:multi_version_tests": {
        "cpu": 2,
        "memory": 268,
    },
    "//max/tests/tests/graph:multi_version_tests_3.10": {
        "cpu": 2,
        "memory": 308,
    },
    "//max/tests/tests/graph:multi_version_tests_3.11": {
        "cpu": 3,
        "memory": 313,
    },
    "//max/tests/tests/graph:multi_version_tests_3.13": {
        "cpu": 2,
        "memory": 291,
    },
    "//max/tests/tests/graph:multi_version_tests_3.14": {
        "cpu": 2,
        "memory": 295,
    },
    "//max/tests/tests/graph:ops/elementwise/test_atanh": {
        "memory": 246,
    },
    "//max/tests/tests/graph:ops/elementwise/test_div": {
        "cpu": 2,
        "memory": 4082,
    },
    "//max/tests/tests/graph:ops/elementwise/test_gelu": {
        "cpu": 2,
        "memory": 256,
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_inf": {
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_nan": {
        "cpu": 2,
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_binary_ops": {
        "memory": 270,
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_not": {
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/elementwise/test_sub": {
        "cpu": 2,
        "memory": 300,
    },
    "//max/tests/tests/graph:ops/reduction/test_argminmax": {
        "memory": 246,
    },
    "//max/tests/tests/graph:ops/test_allgather": {
        "memory": 359,
    },
    "//max/tests/tests/graph:ops/test_allreduce": {
        "cpu": 2,
        "memory": 257,
    },
    "//max/tests/tests/graph:ops/test_argsort": {
        "cpu": 2,
        "memory": 462,
    },
    "//max/tests/tests/graph:ops/test_band_part": {
        "memory": 252,
    },
    "//max/tests/tests/graph:ops/test_broadcast_to": {
        "memory": 267,
    },
    "//max/tests/tests/graph:ops/test_buffer": {
        "cpu": 2,
        "memory": 1414,
    },
    "//max/tests/tests/graph:ops/test_call": {
        "memory": 376,
    },
    "//max/tests/tests/graph:ops/test_cast": {
        "cpu": 2,
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/test_chunk": {
        "cpu": 2,
        "memory": 294,
    },
    "//max/tests/tests/graph:ops/test_complex": {
        "memory": 253,
    },
    "//max/tests/tests/graph:ops/test_concat": {
        "memory": 256,
    },
    "//max/tests/tests/graph:ops/test_conditional": {
        "memory": 265,
    },
    "//max/tests/tests/graph:ops/test_constant": {
        "memory": 428,
    },
    "//max/tests/tests/graph:ops/test_conv": {
        "memory": 259,
    },
    "//max/tests/tests/graph:ops/test_conv3d": {
        "cpu": 2,
        "memory": 291,
    },
    "//max/tests/tests/graph:ops/test_conv_transpose": {
        "memory": 316,
    },
    "//max/tests/tests/graph:ops/test_cumsum": {
        "memory": 259,
    },
    "//max/tests/tests/graph:ops/test_custom": {
        "memory": 839,
    },
    "//max/tests/tests/graph:ops/test_device_chains_collectives": {
        "cpu": 2,
        "memory": 251,
    },
    "//max/tests/tests/graph:ops/test_flatten": {
        "cpu": 2,
        "memory": 270,
    },
    "//max/tests/tests/graph:ops/test_fold": {
        "memory": 470,
    },
    "//max/tests/tests/graph:ops/test_gather": {
        "cpu": 2,
        "memory": 291,
    },
    "//max/tests/tests/graph:ops/test_hann_window": {
        "cpu": 2,
        "memory": 244,
    },
    "//max/tests/tests/graph:ops/test_irfft": {
        "memory": 693,
    },
    "//max/tests/tests/graph:ops/test_layer_norm": {
        "memory": 278,
    },
    "//max/tests/tests/graph:ops/test_linalg": {
        "cpu": 2,
        "memory": 382,
    },
    "//max/tests/tests/graph:ops/test_min_max_overloads": {
        "memory": 253,
    },
    "//max/tests/tests/graph:ops/test_nonzero": {
        "memory": 261,
    },
    "//max/tests/tests/graph:ops/test_outer": {
        "cpu": 2,
        "memory": 290,
    },
    "//max/tests/tests/graph:ops/test_pad": {
        "memory": 268,
    },
    "//max/tests/tests/graph:ops/test_permute": {
        "cpu": 2,
        "memory": 251,
    },
    "//max/tests/tests/graph:ops/test_quantized": {
        "memory": 472,
    },
    "//max/tests/tests/graph:ops/test_random": {
        "memory": 311,
    },
    "//max/tests/tests/graph:ops/test_range": {
        "memory": 313,
    },
    "//max/tests/tests/graph:ops/test_rebind": {
        "memory": 252,
    },
    "//max/tests/tests/graph:ops/test_reduction": {
        "memory": 281,
    },
    "//max/tests/tests/graph:ops/test_repeat_interleave": {
        "memory": 469,
    },
    "//max/tests/tests/graph:ops/test_reshape": {
        "memory": 300,
    },
    "//max/tests/tests/graph:ops/test_resize": {
        "cpu": 2,
        "memory": 246,
    },
    "//max/tests/tests/graph:ops/test_scatter": {
        "cpu": 2,
        "memory": 299,
    },
    "//max/tests/tests/graph:ops/test_shape_to_tensor": {
        "cpu": 2,
        "memory": 259,
    },
    "//max/tests/tests/graph:ops/test_slice": {
        "memory": 326,
    },
    "//max/tests/tests/graph:ops/test_split": {
        "memory": 252,
    },
    "//max/tests/tests/graph:ops/test_stack": {
        "cpu": 2,
        "memory": 314,
    },
    "//max/tests/tests/graph:ops/test_tile": {
        "cpu": 2,
        "memory": 269,
    },
    "//max/tests/tests/graph:ops/test_top_k": {
        "memory": 243,
    },
    "//max/tests/tests/graph:ops/test_transfer": {
        "cpu": 2,
        "memory": 243,
    },
    "//max/tests/tests/graph:ops/test_transpose": {
        "cpu": 2,
        "memory": 288,
    },
    "//max/tests/tests/graph:ops/test_where": {
        "memory": 272,
    },
    "//max/tests/tests/graph:ops/test_while_loop": {
        "cpu": 2,
        "memory": 268,
    },
    "//max/tests/tests/graph:test_debug": {
        "cpu": 2,
        "memory": 294,
    },
    "//max/tests/tests/graph:test_device_ref": {
        "cpu": 2,
        "memory": 247,
    },
    "//max/tests/tests/graph:test_dialects": {
        "memory": 241,
    },
    "//max/tests/tests/graph:test_dtype_promotion": {
        "memory": 328,
    },
    "//max/tests/tests/graph:test_graph_value": {
        "cpu": 2,
        "memory": 311,
    },
    "//max/tests/tests/graph:test_non_contiguous_tensors": {
        "cpu": 2,
        "memory": 1012,
    },
    "//max/tests/tests/graph:test_shapes": {
        "cpu": 2,
        "memory": 272,
    },
    "//max/tests/tests/graph:test_sharding_strategy": {
        "cpu": 2,
        "memory": 268,
    },
    "//max/tests/tests/graph:test_squeeze": {
        "cpu": 2,
        "memory": 270,
    },
    "//max/tests/tests/graph:test_tensor_value": {
        "cpu": 3,
        "memory": 316,
    },
    "//max/tests/tests/graph:test_type": {
        "cpu": 2,
        "memory": 430,
    },
    "//max/tests/tests/graph:test_type_no_context": {
        "memory": 244,
    },
    "//max/tests/tests/graph:test_weight": {
        "memory": 266,
    },
    "//max/tests/tests/graph:utils/test_load_gguf": {
        "memory": 247,
    },
    "//max/tests/tests/graph:utils/test_load_safetensors": {
        "memory": 288,
    },
    "//max/tests/tests/kv_cache:test_attention": {
        "cpu": 2,
    },
    "//max/tests/tests/kv_cache:test_fp4_matmul": {
        "cpu": 2,
        "memory": 735,
    },
    "//max/tests/tests/kv_cache:test_fp8_matmul": {
        "cpu": 2,
        "memory": 620,
    },
    "//max/tests/tests/mojo-importer:mojo-importer": {
        "cpu": 3,
        "memory": 626,
    },
    "//max/tests/tests/nn:test_conv": {
        "cpu": 3,
        "memory": 172,
    },
    "//max/tests/tests/nn:test_layer_norm": {
        "cpu": 3,
        "memory": 115,
    },
    "//max/tests/tests/nn:test_linear": {
        "cpu": 4,
        "memory": 149,
    },
    "//max/tests/tests/nn:test_module": {
        "memory": 1549,
    },
    "//max/tests/tests/nn:test_rms_norm": {
        "cpu": 2,
        "memory": 1562,
    },
    "//max/tests/tests/nn:test_sampling": {
        "cpu": 2,
        "memory": 5336,
    },
    "//max/tests/tests/nn:test_state_dict": {
        "cpu": 4,
        "memory": 109,
    },
    "//max/tests/tests/nn:test_tensor_parallel_linear": {
        "cpu": 4,
    },
    "//max/tests/tests/pipelines/internvl:test_embedding_merge": {
        "cpu": 2,
        "memory": 749,
    },
    "//max/tests/tests/pipelines/internvl:test_embeddings": {
        "cpu": 2,
        "memory": 662,
    },
    "//max/tests/tests/pipelines/lib:test_audio_generation_config": {
        "cpu": 2,
        "memory": 704,
    },
    "//max/tests/tests/pipelines/lib:test_max_config_basic": {
        "cpu": 2,
        "memory": 662,
    },
    "//max/tests/tests/pipelines/lib:test_max_config_inheritance": {
        "cpu": 2,
        "memory": 689,
    },
    "//max/tests/tests/pipelines:test_internvl_weight_adapters": {
        "cpu": 2,
        "memory": 841,
    },
    "//max/tests/tests/pipelines:test_parse_float8_config": {
        "memory": 828,
    },
    "//max/tests/tests/profiler:tests": {
        "cpu": 4,
    },
    "//max/tests/tests/serve/recordreplay:test_replay": {
        "cpu": 6,
        "memory": 134,
    },
    "//max/tests/tests/serve/recordreplay:test_replay_estimation": {
        "cpu": 6,
    },
    "//max/tests/tests/serve/scheduler:test_di": {
        "cpu": 3,
        "memory": 2831,
    },
    "//max/tests/tests/serve/scheduler:test_paged_scheduler": {
        "cpu": 2,
        "memory": 3845,
    },
    "//max/tests/tests/serve/scheduler:test_queues": {
        "memory": 143,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler": {
        "cpu": 2,
        "memory": 652,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_config": {
        "cpu": 2,
        "memory": 634,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_metrics": {
        "cpu": 2,
        "memory": 660,
    },
    "//max/tests/tests/serve/scheduler:test_text_batch_constructor": {
        "cpu": 3,
        "memory": 649,
    },
    "//max/tests/tests/serve/scheduler:test_token_budget": {
        "cpu": 2,
        "memory": 685,
    },
    "//max/tests/tests/serve/scheduler:test_tts_scheduler": {
        "cpu": 2,
        "memory": 1988,
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline": {
        "cpu": 2,
        "memory": 804,
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline_sampling_params": {
        "cpu": 2,
        "memory": 830,
    },
    "//max/tests/tests/serve:pipelines/test_stop_detection": {
        "memory": 864,
    },
    "//max/tests/tests/serve:test_async_queue": {
        "memory": 813,
    },
    "//max/tests/tests/serve:test_file_uri": {
        "cpu": 2,
        "memory": 837,
    },
    "//max/tests/tests/serve:test_kserve_routes": {
        "cpu": 2,
        "memory": 832,
    },
    "//max/tests/tests/serve:test_llm": {
        "memory": 977,
    },
    "//max/tests/tests/serve:test_lora_integration": {
        "cpu": 2,
        "memory": 814,
    },
    "//max/tests/tests/serve:test_metrics": {
        "cpu": 2,
        "memory": 793,
    },
    "//max/tests/tests/serve:test_multiprocessing": {
        "cpu": 2,
        "memory": 806,
    },
    "//max/tests/tests/serve:test_openai_request": {
        "cpu": 2,
        "memory": 962,
    },
    "//max/tests/tests/serve:test_openai_routes": {
        "cpu": 2,
        "memory": 1054,
    },
    "//max/tests/tests/serve:test_openai_stream": {
        "cpu": 2,
        "memory": 978,
    },
    "//max/tests/tests/serve:test_reset_prefix_cache": {
        "memory": 978,
    },
    "//max/tests/tests/serve:test_routes": {
        "cpu": 2,
        "memory": 982,
    },
    "//max/tests/tests/serve:test_socket_settings": {
        "cpu": 2,
        "memory": 836,
    },
    "//max/tests/tests/serve:test_stopwatch": {
        "memory": 958,
    },
    "//max/tests/tests/serve:test_telemetry_worker": {
        "memory": 839,
    },
    "//max/tests/tests/support:tests": {
        "cpu": 2,
        "memory": 436,
    },
    "//max/tests/tests/torch:tests": {
        "cpu": 2,
        "memory": 6994,
    },
    "//max/tests/tests:test_generated_dialects": {
        "cpu": 8,
        "memory": 1849,
    },
    "//max/tests/tests:test_passes": {
        "cpu": 6,
        "memory": 2639,
    },
    "//max/tests/tests:test_realization_context": {
        "cpu": 8,
        "memory": 18819,
    },
    "//max/tests/tests:test_support": {
        "cpu": 8,
        "memory": 1951,
    },
    "//max/tests/tests:test_tensor": {
        "cpu": 8,
        "memory": 18240,
    },
}
