"""Resource estimates for tests, generated internally by utils/generate_test_resources_report"""
TEST_RESOURCES = {
    "//max/examples/capi:test": {
        "cpu": 2,
        "memory": 812,
    },
    "//max/examples/custom-graph-module:main_test": {
        "cpu": 5,
    },
    "//max/examples/custom_ops:addition.example-test": {
        "cpu": 2,
        "memory": 2828,
    },
    "//max/examples/custom_ops:histogram.example-test": {
        "cpu": 3,
        "memory": 2838,
    },
    "//max/examples/custom_ops:image_pipeline.example-test": {
        "cpu": 2,
        "memory": 2851,
    },
    "//max/examples/custom_ops:mandelbrot.example-test": {
        "cpu": 3,
        "memory": 2882,
    },
    "//max/examples/custom_ops:parametric_addition.example-test": {
        "cpu": 2,
        "memory": 2847,
    },
    "//max/examples/custom_ops:top_k.example-test": {
        "cpu": 3,
        "memory": 2848,
    },
    "//max/examples/custom_ops:vector_addition.example-test": {
        "cpu": 2,
        "memory": 2768,
    },
    "//max/examples/internal/transfer_engine:test_send_recv": {
        "cpu": 2,
        "memory": 812,
    },
    "//max/examples/max-graph:addition_test": {
        "cpu": 2,
        "memory": 767,
    },
    "//max/examples/offline-inference:basic_test": {
        "cpu": 6,
        "memory": 48219,
    },
    "//max/examples/pytorch_custom_ops:addition.example-test": {
        "cpu": 3,
        "memory": 2993,
    },
    "//max/examples/pytorch_custom_ops:graph.example-test": {
        "cpu": 2,
        "memory": 1467,
    },
    "//max/examples/pytorch_custom_ops:grayscale.example-test": {
        "cpu": 2,
        "memory": 3352,
    },
    "//max/kernels/benchmarks/autotune:tests/test_kbench": {
        "cpu": 5,
        "memory": 2038,
    },
    "//max/kernels/benchmarks:algorithm/parallelize_overhead.mojo.test": {
        "memory": 800,
    },
    "//max/kernels/test/gpu/linalg:test_matmul_sm100_ptx.mojo.test": {
        "memory": 300,
    },
    "//max/kernels/test/kv_cache:test_mha_mixed_ce_tg.mojo.test": {
        "memory": 193,
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
        "memory": 458,
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
        "memory": 183,
    },
    "//max/kernels/test/quantization:test_qmatmul_k.mojo.test": {
        "cpu": 6,
    },
    "//max/tests/integration/API/python/graph:test_matmul_packed": {
        "cpu": 2,
        "memory": 2268,
    },
    "//max/tests/integration/API/python/graph:test_reduce_add": {
        "cpu": 2,
        "memory": 953,
    },
    "//max/tests/integration/API/python/interfaces:test_hash_image": {
        "cpu": 5,
        "memory": 300,
    },
    "//max/tests/integration/API/python/interfaces:test_queue": {
        "cpu": 3,
        "memory": 300,
    },
    "//max/tests/integration/API/python/interfaces:test_serialization": {
        "cpu": 5,
        "memory": 300,
    },
    "//max/tests/integration/API/python/interfaces:test_tokens": {
        "cpu": 4,
        "memory": 300,
    },
    "//max/tests/integration/API/python/nn/module_v3:norm/test_rms_norm": {
        "cpu": 8,
        "memory": 17321,
    },
    "//max/tests/integration/API/python/nn/module_v3:rope/test_rope": {
        "cpu": 6,
        "memory": 27722,
    },
    "//max/tests/integration/API/python/nn/module_v3:rope/test_yarn": {
        "cpu": 3,
        "memory": 13737,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_embedding": {
        "cpu": 5,
        "memory": 11005,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_linear": {
        "cpu": 8,
        "memory": 10338,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_module": {
        "cpu": 8,
        "memory": 32579,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_module_gpt2": {
        "cpu": 4,
        "memory": 16676,
    },
    "//max/tests/integration/API/python/nn/module_v3:test_sequential": {
        "cpu": 4,
        "memory": 4631,
    },
    "//max/tests/integration/API/python/tensor:test_arange": {
        "cpu": 8,
        "memory": 39657,
    },
    "//max/tests/integration/API/python/tensor:test_functional_binary": {
        "cpu": 8,
        "memory": 34756,
    },
    "//max/tests/integration/API/python/tensor:test_functional_custom": {
        "cpu": 8,
        "memory": 3770,
    },
    "//max/tests/integration/API/python/tensor:test_functional_other": {
        "cpu": 8,
        "memory": 79132,
    },
    "//max/tests/integration/API/python/tensor:test_functional_reduction": {
        "cpu": 8,
        "memory": 10288,
    },
    "//max/tests/integration/API/python/tensor:test_functional_unary": {
        "cpu": 8,
        "memory": 47127,
    },
    "//max/tests/integration/API/python/tensor:test_random": {
        "cpu": 8,
        "memory": 23106,
    },
    "//max/tests/integration/API/python/tensor:test_tensor": {
        "cpu": 8,
        "memory": 113387,
    },
    "//max/tests/integration/API/python:test_load_library": {
        "cpu": 4,
    },
    "//max/tests/integration/API/python:test_load_library_3.10": {
        "cpu": 3,
    },
    "//max/tests/integration/API/python:test_load_library_3.11": {
        "cpu": 5,
    },
    "//max/tests/integration/API/python:test_load_library_3.13": {
        "cpu": 6,
    },
    "//max/tests/integration/API/python:tests-fail-weight-loading": {
        "cpu": 3,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_causal_attention_mask": {
        "memory": 300,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_causal_attention_mask_with_alibi": {
        "memory": 398,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_collate_batch": {
        "cpu": 3,
        "memory": 300,
    },
    "//max/tests/integration/pipelines/python/dataprocessing:test_max_tokens_to_generate": {
        "cpu": 4,
        "memory": 300,
    },
    "//max/tests/integration/pipelines/python/kv_cache/attention:attention_no_opaque_tests": {
        "cpu": 2,
        "memory": 3521,
    },
    "//max/tests/integration/pipelines/python/kv_cache/attention:attention_tests": {
        "cpu": 2,
        "memory": 4292,
    },
    "//max/tests/integration/pipelines/python/kv_cache/transfer_engine:test_notification_latency": {
        "cpu": 3,
        "memory": 1728,
    },
    "//max/tests/integration/pipelines/python/kv_cache/transfer_engine:test_send_recv": {
        "cpu": 2,
        "memory": 725,
    },
    "//max/tests/integration/pipelines/python/kv_cache:embedding": {
        "cpu": 8,
        "memory": 19446,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_kv_cache_matmul": {
        "cpu": 2,
        "memory": 4466,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_memory_estimation": {
        "cpu": 2,
        "memory": 247,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_prefix_caching": {
        "memory": 2255,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_print_kv_cache": {
        "cpu": 2,
        "memory": 9310,
    },
    "//max/tests/integration/pipelines/python/kv_cache:test_rms_norm_key_cache": {
        "cpu": 2,
        "memory": 3634,
    },
    "//max/tests/integration/pipelines/python/mistral3:tests": {
        "memory": 676,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_block_hasher": {
        "cpu": 2,
        "memory": 801,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_cache_params": {
        "cpu": 2,
        "memory": 247,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_data_parallelism_utils": {
        "memory": 248,
    },
    "//max/tests/integration/pipelines/python/nn/kv_cache:test_kv_cache_manager": {
        "cpu": 2,
        "memory": 2358,
    },
    "//max/tests/integration/pipelines/python/nn/norm:norm_tests": {
        "cpu": 2,
        "memory": 1525,
    },
    "//max/tests/integration/pipelines/python/nn:test_conv": {
        "cpu": 8,
        "memory": 14674,
    },
    "//max/tests/integration/pipelines/python/nn:test_identity": {
        "cpu": 8,
        "memory": 6914,
    },
    "//max/tests/integration/pipelines/python/nn:test_layer_hook": {
        "cpu": 8,
        "memory": 7111,
    },
    "//max/tests/integration/pipelines/python/nn:test_mlp": {
        "cpu": 8,
        "memory": 50641,
    },
    "//max/tests/integration/pipelines/python/nn:test_print_hook": {
        "cpu": 8,
        "memory": 7769,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_compute_log_probabilities": {
        "cpu": 2,
        "memory": 1433,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_lora_graph_inputs": {
        "cpu": 2,
        "memory": 815,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_pipeline_lora_sorting": {
        "cpu": 2,
        "memory": 834,
    },
    "//max/tests/integration/pipelines/python/pipelines:test_text_generation_pipeline": {
        "cpu": 2,
        "memory": 1524,
    },
    "//max/tests/integration/pipelines/python/qwen2_5vl:test_compute_scatter_gather_indices": {
        "cpu": 2,
        "memory": 674,
    },
    "//max/tests/integration/pipelines/python/qwen2_5vl:test_vision_functions": {
        "memory": 12509,
    },
    "//max/tests/integration/pipelines/python/qwen3vl:test_vision_functions": {
        "cpu": 2,
        "memory": 808,
    },
    "//max/tests/integration/pipelines/python/whisper:whisper": {
        "cpu": 2,
        "memory": 857,
    },
    "//max/tests/integration/pipelines/python:test_debug_model": {
        "cpu": 2,
        "memory": 822,
    },
    "//max/tests/integration/pipelines/python:test_debug_utils": {
        "cpu": 2,
        "memory": 874,
    },
    "//max/tests/integration/pipelines/python:test_hf_config_overrides": {
        "cpu": 2,
        "memory": 669,
    },
    "//max/tests/integration/pipelines/python:test_hf_repo_lock": {
        "cpu": 2,
        "memory": 875,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli": {
        "cpu": 5,
        "memory": 4797,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli_help": {
        "cpu": 2,
        "memory": 670,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli_json_lightweight": {
        "cpu": 2,
        "memory": 675,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_cli_lightweight": {
        "cpu": 2,
        "memory": 686,
    },
    "//max/tests/integration/pipelines/python:test_pipelines_lm_eval": {
        "cpu": 4,
        "memory": 6567,
    },
    "//max/tests/integration/serve/kvcache_agent:tests": {
        "cpu": 3,
    },
    "//max/tests/integration/serve:test_sagemaker_cpu": {
        "cpu": 2,
        "memory": 1048,
    },
    "//max/tests/integration/serve:test_stop_cpu": {
        "cpu": 2,
        "memory": 1073,
    },
    "//max/tests/internal/examples:test_load_balancer": {
        "cpu": 2,
        "memory": 116,
    },
    "//max/tests/internal/telemetry:tests": {
        "cpu": 4,
    },
    "//max/tests/tests/_core_mojo:tests": {
        "cpu": 2,
        "memory": 611,
    },
    "//max/tests/tests/driver:test_device": {
        "cpu": 2,
        "memory": 220,
    },
    "//max/tests/tests/driver:test_driver": {
        "memory": 220,
    },
    "//max/tests/tests/driver:test_tensor": {
        "cpu": 2,
        "memory": 222,
    },
    "//max/tests/tests/entrypoints:tests": {
        "cpu": 2,
        "memory": 694,
    },
    "//max/tests/tests/graph:multi_version_tests": {
        "cpu": 2,
        "memory": 283,
    },
    "//max/tests/tests/graph:multi_version_tests_3.10": {
        "cpu": 2,
        "memory": 270,
    },
    "//max/tests/tests/graph:multi_version_tests_3.11": {
        "cpu": 3,
        "memory": 308,
    },
    "//max/tests/tests/graph:multi_version_tests_3.13": {
        "cpu": 2,
        "memory": 320,
    },
    "//max/tests/tests/graph:ops/elementwise/test_atanh": {
        "cpu": 2,
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/elementwise/test_div": {
        "cpu": 2,
        "memory": 2975,
    },
    "//max/tests/tests/graph:ops/elementwise/test_gelu": {
        "cpu": 2,
        "memory": 287,
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_inf": {
        "memory": 242,
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_nan": {
        "cpu": 2,
        "memory": 244,
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_binary_ops": {
        "memory": 256,
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_not": {
        "cpu": 2,
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/elementwise/test_sub": {
        "cpu": 2,
        "memory": 302,
    },
    "//max/tests/tests/graph:ops/reduction/test_argminmax": {
        "cpu": 2,
        "memory": 246,
    },
    "//max/tests/tests/graph:ops/test_allgather": {
        "cpu": 2,
        "memory": 278,
    },
    "//max/tests/tests/graph:ops/test_allreduce": {
        "cpu": 2,
        "memory": 258,
    },
    "//max/tests/tests/graph:ops/test_argsort": {
        "cpu": 2,
        "memory": 512,
    },
    "//max/tests/tests/graph:ops/test_band_part": {
        "memory": 252,
    },
    "//max/tests/tests/graph:ops/test_broadcast_to": {
        "memory": 263,
    },
    "//max/tests/tests/graph:ops/test_buffer": {
        "cpu": 2,
        "memory": 1781,
    },
    "//max/tests/tests/graph:ops/test_call": {
        "cpu": 3,
        "memory": 347,
    },
    "//max/tests/tests/graph:ops/test_cast": {
        "cpu": 2,
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/test_chunk": {
        "memory": 266,
    },
    "//max/tests/tests/graph:ops/test_complex": {
        "cpu": 2,
        "memory": 252,
    },
    "//max/tests/tests/graph:ops/test_concat": {
        "memory": 255,
    },
    "//max/tests/tests/graph:ops/test_conditional": {
        "cpu": 2,
        "memory": 266,
    },
    "//max/tests/tests/graph:ops/test_constant": {
        "memory": 433,
    },
    "//max/tests/tests/graph:ops/test_conv": {
        "memory": 258,
    },
    "//max/tests/tests/graph:ops/test_conv3d": {
        "cpu": 2,
        "memory": 275,
    },
    "//max/tests/tests/graph:ops/test_conv_transpose": {
        "cpu": 2,
        "memory": 274,
    },
    "//max/tests/tests/graph:ops/test_cumsum": {
        "cpu": 2,
        "memory": 285,
    },
    "//max/tests/tests/graph:ops/test_custom": {
        "cpu": 2,
        "memory": 1335,
    },
    "//max/tests/tests/graph:ops/test_device_chains_collectives": {
        "cpu": 2,
        "memory": 242,
    },
    "//max/tests/tests/graph:ops/test_flatten": {
        "memory": 273,
    },
    "//max/tests/tests/graph:ops/test_fold": {
        "memory": 553,
    },
    "//max/tests/tests/graph:ops/test_gather": {
        "memory": 289,
    },
    "//max/tests/tests/graph:ops/test_hann_window": {
        "cpu": 2,
        "memory": 245,
    },
    "//max/tests/tests/graph:ops/test_irfft": {
        "memory": 819,
    },
    "//max/tests/tests/graph:ops/test_layer_norm": {
        "memory": 248,
    },
    "//max/tests/tests/graph:ops/test_linalg": {
        "cpu": 2,
        "memory": 374,
    },
    "//max/tests/tests/graph:ops/test_min_max_overloads": {
        "memory": 253,
    },
    "//max/tests/tests/graph:ops/test_nonzero": {
        "cpu": 2,
        "memory": 264,
    },
    "//max/tests/tests/graph:ops/test_outer": {
        "cpu": 2,
        "memory": 278,
    },
    "//max/tests/tests/graph:ops/test_pad": {
        "cpu": 2,
        "memory": 250,
    },
    "//max/tests/tests/graph:ops/test_permute": {
        "memory": 251,
    },
    "//max/tests/tests/graph:ops/test_quantized": {
        "cpu": 2,
        "memory": 555,
    },
    "//max/tests/tests/graph:ops/test_random": {
        "memory": 302,
    },
    "//max/tests/tests/graph:ops/test_range": {
        "cpu": 2,
        "memory": 360,
    },
    "//max/tests/tests/graph:ops/test_rebind": {
        "cpu": 2,
        "memory": 248,
    },
    "//max/tests/tests/graph:ops/test_reduction": {
        "memory": 279,
    },
    "//max/tests/tests/graph:ops/test_repeat_interleave": {
        "memory": 516,
    },
    "//max/tests/tests/graph:ops/test_reshape": {
        "memory": 290,
    },
    "//max/tests/tests/graph:ops/test_resize": {
        "memory": 247,
    },
    "//max/tests/tests/graph:ops/test_scatter": {
        "cpu": 2,
        "memory": 283,
    },
    "//max/tests/tests/graph:ops/test_shape_to_tensor": {
        "cpu": 2,
        "memory": 267,
    },
    "//max/tests/tests/graph:ops/test_slice": {
        "cpu": 2,
        "memory": 334,
    },
    "//max/tests/tests/graph:ops/test_split": {
        "memory": 252,
    },
    "//max/tests/tests/graph:ops/test_stack": {
        "cpu": 2,
        "memory": 318,
    },
    "//max/tests/tests/graph:ops/test_tile": {
        "cpu": 2,
        "memory": 268,
    },
    "//max/tests/tests/graph:ops/test_top_k": {
        "cpu": 2,
        "memory": 243,
    },
    "//max/tests/tests/graph:ops/test_transfer": {
        "cpu": 2,
        "memory": 242,
    },
    "//max/tests/tests/graph:ops/test_transpose": {
        "cpu": 2,
        "memory": 296,
    },
    "//max/tests/tests/graph:ops/test_where": {
        "cpu": 2,
        "memory": 273,
    },
    "//max/tests/tests/graph:ops/test_while_loop": {
        "memory": 264,
    },
    "//max/tests/tests/graph:test_debug": {
        "cpu": 2,
        "memory": 281,
    },
    "//max/tests/tests/graph:test_device_ref": {
        "cpu": 2,
        "memory": 242,
    },
    "//max/tests/tests/graph:test_dialects": {
        "memory": 241,
    },
    "//max/tests/tests/graph:test_dtype_promotion": {
        "cpu": 2,
        "memory": 339,
    },
    "//max/tests/tests/graph:test_graph_value": {
        "memory": 276,
    },
    "//max/tests/tests/graph:test_non_contiguous_tensors": {
        "memory": 827,
    },
    "//max/tests/tests/graph:test_shapes": {
        "memory": 243,
    },
    "//max/tests/tests/graph:test_sharding_strategy": {
        "cpu": 2,
        "memory": 262,
    },
    "//max/tests/tests/graph:test_squeeze": {
        "memory": 266,
    },
    "//max/tests/tests/graph:test_tensor_value": {
        "cpu": 2,
        "memory": 307,
    },
    "//max/tests/tests/graph:test_type": {
        "memory": 422,
    },
    "//max/tests/tests/graph:test_type_no_context": {
        "memory": 244,
    },
    "//max/tests/tests/graph:test_weight": {
        "memory": 263,
    },
    "//max/tests/tests/graph:utils/test_load_gguf": {
        "cpu": 2,
        "memory": 244,
    },
    "//max/tests/tests/graph:utils/test_load_safetensors": {
        "cpu": 2,
        "memory": 253,
    },
    "//max/tests/tests/kv_cache:test_attention": {
        "cpu": 3,
    },
    "//max/tests/tests/kv_cache:test_fp4_matmul": {
        "memory": 876,
    },
    "//max/tests/tests/kv_cache:test_fp8_matmul": {
        "memory": 705,
    },
    "//max/tests/tests/mojo-importer:mojo-importer": {
        "cpu": 3,
        "memory": 635,
    },
    "//max/tests/tests/nn:test_conv": {
        "cpu": 4,
    },
    "//max/tests/tests/nn:test_layer_norm": {
        "cpu": 5,
    },
    "//max/tests/tests/nn:test_linear": {
        "cpu": 4,
        "memory": 123,
    },
    "//max/tests/tests/nn:test_module": {
        "cpu": 2,
        "memory": 781,
    },
    "//max/tests/tests/nn:test_rms_norm": {
        "memory": 1617,
    },
    "//max/tests/tests/nn:test_sampling": {
        "cpu": 2,
        "memory": 3187,
    },
    "//max/tests/tests/nn:test_state_dict": {
        "cpu": 2,
    },
    "//max/tests/tests/nn:test_tensor_parallel_linear": {
        "cpu": 4,
    },
    "//max/tests/tests/pipelines/internvl:test_embedding_merge": {
        "cpu": 2,
        "memory": 651,
    },
    "//max/tests/tests/pipelines/internvl:test_embeddings": {
        "cpu": 2,
        "memory": 676,
    },
    "//max/tests/tests/pipelines/lib:test_audio_generation_config": {
        "cpu": 2,
        "memory": 837,
    },
    "//max/tests/tests/pipelines/lib:test_max_config_basic": {
        "cpu": 2,
        "memory": 845,
    },
    "//max/tests/tests/pipelines/lib:test_max_config_inheritance": {
        "cpu": 2,
        "memory": 836,
    },
    "//max/tests/tests/pipelines:test_internvl_weight_adapters": {
        "cpu": 2,
        "memory": 810,
    },
    "//max/tests/tests/pipelines:test_parse_float8_config": {
        "cpu": 2,
        "memory": 930,
    },
    "//max/tests/tests/profiler:tests": {
        "cpu": 4,
    },
    "//max/tests/tests/serve/recordreplay:test_replay": {
        "cpu": 6,
        "memory": 103,
    },
    "//max/tests/tests/serve/recordreplay:test_replay_estimation": {
        "cpu": 7,
        "memory": 101,
    },
    "//max/tests/tests/serve/scheduler:test_di": {
        "cpu": 3,
        "memory": 3488,
    },
    "//max/tests/tests/serve/scheduler:test_paged_scheduler": {
        "cpu": 2,
        "memory": 4233,
    },
    "//max/tests/tests/serve/scheduler:test_queues": {
        "memory": 142,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler": {
        "cpu": 2,
        "memory": 657,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_config": {
        "cpu": 2,
        "memory": 719,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_metrics": {
        "cpu": 2,
        "memory": 644,
    },
    "//max/tests/tests/serve/scheduler:test_text_batch_constructor": {
        "cpu": 2,
        "memory": 649,
    },
    "//max/tests/tests/serve/scheduler:test_token_budget": {
        "cpu": 2,
        "memory": 618,
    },
    "//max/tests/tests/serve/scheduler:test_tts_scheduler": {
        "cpu": 2,
        "memory": 1936,
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline": {
        "cpu": 2,
        "memory": 873,
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline_sampling_params": {
        "cpu": 2,
        "memory": 830,
    },
    "//max/tests/tests/serve:pipelines/test_stop_detection": {
        "cpu": 2,
        "memory": 820,
    },
    "//max/tests/tests/serve:test_async_queue": {
        "memory": 782,
    },
    "//max/tests/tests/serve:test_file_uri": {
        "cpu": 2,
        "memory": 810,
    },
    "//max/tests/tests/serve:test_kserve_routes": {
        "cpu": 2,
        "memory": 840,
    },
    "//max/tests/tests/serve:test_llm": {
        "memory": 980,
    },
    "//max/tests/tests/serve:test_lora_integration": {
        "cpu": 2,
        "memory": 831,
    },
    "//max/tests/tests/serve:test_metrics": {
        "cpu": 2,
        "memory": 790,
    },
    "//max/tests/tests/serve:test_multiprocessing": {
        "memory": 801,
    },
    "//max/tests/tests/serve:test_openai_request": {
        "cpu": 2,
        "memory": 839,
    },
    "//max/tests/tests/serve:test_openai_routes": {
        "memory": 1054,
    },
    "//max/tests/tests/serve:test_openai_stream": {
        "cpu": 2,
        "memory": 974,
    },
    "//max/tests/tests/serve:test_reset_prefix_cache": {
        "memory": 977,
    },
    "//max/tests/tests/serve:test_routes": {
        "cpu": 2,
        "memory": 973,
    },
    "//max/tests/tests/serve:test_socket_settings": {
        "cpu": 2,
        "memory": 841,
    },
    "//max/tests/tests/serve:test_stopwatch": {
        "cpu": 2,
        "memory": 824,
    },
    "//max/tests/tests/serve:test_telemetry_worker": {
        "cpu": 2,
        "memory": 831,
    },
    "//max/tests/tests/support:tests": {
        "cpu": 2,
        "memory": 424,
    },
    "//max/tests/tests/torch:tests": {
        "cpu": 2,
        "memory": 8447,
    },
    "//max/tests/tests:test_generated_dialects": {
        "cpu": 8,
        "memory": 1805,
    },
    "//max/tests/tests:test_passes": {
        "cpu": 8,
        "memory": 2952,
    },
    "//max/tests/tests:test_realization_context": {
        "cpu": 8,
        "memory": 17104,
    },
    "//max/tests/tests:test_support": {
        "cpu": 8,
        "memory": 2016,
    },
    "//max/tests/tests:test_tensor": {
        "cpu": 8,
        "memory": 16441,
    },
}
