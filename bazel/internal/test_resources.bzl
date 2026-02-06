"""Resource estimates for tests, generated internally by utils/generate_test_resources_report"""
TEST_RESOURCES = {
    "//max/examples/capi:test": {
        "cpu": 2,
        "memory": 1464,
    },
    "//max/examples/custom-graph-module:main_test": {
        "cpu": 7,
    },
    "//max/examples/custom_ops:addition.example-test": {
        "cpu": 2,
        "memory": 1409,
    },
    "//max/examples/custom_ops:histogram.example-test": {
        "cpu": 2,
        "memory": 1445,
    },
    "//max/examples/custom_ops:image_pipeline.example-test": {
        "cpu": 2,
        "memory": 1494,
    },
    "//max/examples/custom_ops:mandelbrot.example-test": {
        "cpu": 2,
        "memory": 1422,
    },
    "//max/examples/custom_ops:parametric_addition.example-test": {
        "cpu": 2,
        "memory": 1402,
    },
    "//max/examples/custom_ops:top_k.example-test": {
        "cpu": 2,
        "memory": 1939,
    },
    "//max/examples/custom_ops:vector_addition.example-test": {
        "cpu": 2,
        "memory": 1261,
    },
    "//max/examples/max-graph:addition_test": {
        "cpu": 2,
        "memory": 1199,
    },
    "//max/examples/pytorch_custom_ops:addition.example-test": {
        "cpu": 2,
        "memory": 1822,
    },
    "//max/examples/pytorch_custom_ops:graph.example-test": {
        "memory": 3686,
    },
    "//max/examples/pytorch_custom_ops:grayscale.example-test": {
        "cpu": 2,
        "memory": 1888,
    },
    "//max/kernels/benchmarks/autotune:tests/test_kbench": {
        "cpu": 3,
        "memory": 2262,
    },
    "//max/kernels/benchmarks:algorithm/parallelize_overhead.mojo.test": {
        "memory": 960,
    },
    "//max/kernels/test/gpu/linalg:test_matmul_sm100_ptx.mojo.test": {
        "memory": 360,
    },
    "//max/kernels/test/kv_cache:test_mha_mixed_ce_tg.mojo.test": {
        "memory": 232,
    },
    "//max/kernels/test/linalg:test_gemv.mojo.test": {
        "cpu": 8,
        "memory": 233,
    },
    "//max/kernels/test/linalg:test_neon_dotprod_intrinsics.mojo.test": {
        "memory": 360,
    },
    "//max/kernels/test/linalg:test_neon_matmul_intrinsics.mojo.test": {
        "memory": 360,
    },
    "//max/kernels/test/linalg:test_vnni_intrinsics.mojo.test": {
        "memory": 488,
    },
    "//max/kernels/test/nn:test_conv1d.mojo.test": {
        "memory": 130,
    },
    "//max/kernels/test/nn:test_direct_conv.mojo.test": {
        "memory": 808,
    },
    "//max/kernels/test/nn:test_toppminp.mojo.test": {
        "memory": 209,
    },
    "//max/kernels/test/quantization:test_qmatmul_k.mojo.test": {
        "cpu": 4,
    },
    "//max/tests/integration/graph:test_matmul_packed": {
        "memory": 5897,
    },
    "//max/tests/integration/graph:test_reduce_add": {
        "memory": 1890,
    },
    "//max/tests/integration/interfaces:test_hash_image": {
        "cpu": 6,
        "memory": 360,
    },
    "//max/tests/integration/interfaces:test_queue": {
        "cpu": 5,
        "memory": 360,
    },
    "//max/tests/integration/interfaces:test_serialization": {
        "cpu": 5,
        "memory": 360,
    },
    "//max/tests/integration/interfaces:test_tokens": {
        "cpu": 5,
        "memory": 360,
    },
    "//max/tests/integration/interfaces:text_generation/test_text_generation_request": {
        "cpu": 5,
        "memory": 360,
    },
    "//max/tests/integration/nn/module_v3:norm/test_rms_norm": {
        "cpu": 3,
        "memory": 21602,
    },
    "//max/tests/integration/nn/module_v3:rope/test_rope": {
        "cpu": 4,
        "memory": 29830,
    },
    "//max/tests/integration/nn/module_v3:test_embedding": {
        "cpu": 4,
        "memory": 16103,
    },
    "//max/tests/integration/nn/module_v3:test_linear": {
        "cpu": 3,
        "memory": 15871,
    },
    "//max/tests/integration/nn/module_v3:test_module": {
        "cpu": 7,
        "memory": 42461,
    },
    "//max/tests/integration/nn/module_v3:test_sequential": {
        "cpu": 3,
        "memory": 5680,
    },
    "//max/tests/integration/tensor:test_arange": {
        "cpu": 4,
        "memory": 43294,
    },
    "//max/tests/integration/tensor:test_functional_binary": {
        "cpu": 5,
        "memory": 44612,
    },
    "//max/tests/integration/tensor:test_functional_custom": {
        "cpu": 8,
        "memory": 4387,
    },
    "//max/tests/integration/tensor:test_functional_other": {
        "cpu": 7,
        "memory": 117646,
    },
    "//max/tests/integration/tensor:test_functional_reduction": {
        "cpu": 8,
        "memory": 13848,
    },
    "//max/tests/integration/tensor:test_functional_unary": {
        "cpu": 6,
        "memory": 58646,
    },
    "//max/tests/integration/tensor:test_random": {
        "cpu": 5,
        "memory": 29461,
    },
    "//max/tests/integration/tensor:test_tensor_elemwise": {
        "cpu": 7,
        "memory": 111186,
    },
    "//max/tests/integration/tensor:test_tensor_matmul": {
        "cpu": 5,
        "memory": 12228,
    },
    "//max/tests/integration/tensor:test_tensor_repr": {
        "cpu": 7,
        "memory": 42664,
    },
    "//max/tests/integration/unorganized:test_load_library": {
        "cpu": 5,
    },
    "//max/tests/integration/unorganized:test_load_library_3.10": {
        "cpu": 7,
    },
    "//max/tests/integration/unorganized:test_load_library_3.11": {
        "cpu": 7,
    },
    "//max/tests/integration/unorganized:test_load_library_3.13": {
        "cpu": 7,
        "memory": 127,
    },
    "//max/tests/integration/unorganized:test_load_library_3.14": {
        "cpu": 6,
    },
    "//max/tests/integration/unorganized:tests-fail-weight-loading": {
        "cpu": 5,
    },
    "//max/tests/integration/dataprocessing:test_causal_attention_mask": {
        "memory": 360,
    },
    "//max/tests/integration/dataprocessing:test_causal_attention_mask_with_alibi": {
        "memory": 475,
    },
    "//max/tests/integration/dataprocessing:test_collate_batch": {
        "cpu": 2,
        "memory": 360,
    },
    "//max/tests/integration/dataprocessing:test_max_tokens_to_generate": {
        "cpu": 5,
        "memory": 360,
    },
    "//max/tests/integration/kv_cache/attention:attention_no_opaque_tests": {
        "cpu": 2,
        "memory": 4325,
    },
    "//max/tests/integration/kv_cache/attention:attention_tests": {
        "cpu": 2,
        "memory": 6932,
    },
    "//max/tests/integration/kv_cache/transfer_engine:test_notification_latency": {
        "cpu": 2,
        "memory": 2078,
    },
    "//max/tests/integration/kv_cache/transfer_engine:test_send_recv": {
        "cpu": 3,
        "memory": 710,
    },
    "//max/tests/integration/kv_cache:embedding": {
        "cpu": 8,
        "memory": 21692,
    },
    "//max/tests/integration/kv_cache:test_kv_cache_matmul": {
        "cpu": 2,
        "memory": 19446,
    },
    "//max/tests/integration/kv_cache:test_memory_estimation": {
        "cpu": 2,
        "memory": 295,
    },
    "//max/tests/integration/kv_cache:test_prefix_caching": {
        "memory": 2448,
    },
    "//max/tests/integration/kv_cache:test_print_kv_cache": {
        "cpu": 2,
        "memory": 13339,
    },
    "//max/tests/integration/kv_cache:test_rms_norm_key_cache": {
        "cpu": 2,
        "memory": 5705,
    },
    "//max/tests/integration/architectures/mistral3:tests": {
        "cpu": 2,
        "memory": 721,
    },
    "//max/tests/integration/nn/kv_cache:test_block_hasher": {
        "cpu": 2,
        "memory": 875,
    },
    "//max/tests/integration/nn/kv_cache:test_cache_params": {
        "cpu": 2,
        "memory": 296,
    },
    "//max/tests/integration/nn/kv_cache:test_data_parallelism_utils": {
        "memory": 295,
    },
    "//max/tests/integration/nn/kv_cache:test_kv_cache_manager": {
        "memory": 2587,
    },
    "//max/tests/integration/nn/norm:norm_tests": {
        "memory": 3623,
    },
    "//max/tests/integration/nn:test_conv": {
        "cpu": 3,
        "memory": 21245,
    },
    "//max/tests/integration/nn:test_identity": {
        "cpu": 8,
        "memory": 8041,
    },
    "//max/tests/integration/nn:test_layer_hook": {
        "cpu": 8,
        "memory": 8842,
    },
    "//max/tests/integration/nn:test_mlp": {
        "cpu": 8,
        "memory": 55325,
    },
    "//max/tests/integration/nn:test_print_hook": {
        "cpu": 6,
        "memory": 10141,
    },
    "//max/tests/integration/pipelines:test_compute_log_probabilities": {
        "cpu": 2,
        "memory": 2149,
    },
    "//max/tests/integration/pipelines:test_lora_graph_inputs": {
        "cpu": 2,
        "memory": 859,
    },
    "//max/tests/integration/pipelines:test_pipeline_lora_sorting": {
        "cpu": 2,
        "memory": 896,
    },
    "//max/tests/integration/pipelines:test_text_generation_pipeline": {
        "cpu": 2,
        "memory": 2246,
    },
    "//max/tests/integration/architectures/qwen2_5vl:test_compute_scatter_gather_indices": {
        "cpu": 2,
        "memory": 776,
    },
    "//max/tests/integration/architectures/qwen2_5vl:test_vision_functions": {
        "memory": 14933,
    },
    "//max/tests/integration/architectures/qwen3vl:test_vision_functions": {
        "cpu": 2,
        "memory": 935,
    },
    "//max/tests/integration/architectures/whisper:whisper": {
        "cpu": 2,
        "memory": 935,
    },
    "//max/tests/integration/accuracy:test_compare_tensors": {
        "memory": 2370,
    },
    "//max/tests/integration/accuracy:test_debug_model": {
        "cpu": 2,
        "memory": 893,
    },
    "//max/tests/integration/accuracy:test_debug_utils": {
        "cpu": 2,
        "memory": 911,
    },
    "//max/tests/integration/tools:test_hf_config_overrides": {
        "cpu": 2,
        "memory": 743,
    },
    "//max/tests/integration:test_hf_repo_lock": {
        "cpu": 2,
        "memory": 894,
    },
    "//max/tests/integration/cli:test_pipelines_cli_help": {
        "cpu": 2,
        "memory": 775,
    },
    "//max/tests/integration/cli:test_pipelines_cli_json_lightweight": {
        "memory": 748,
    },
    "//max/tests/integration/cli:test_pipelines_cli_lightweight": {
        "cpu": 2,
        "memory": 778,
    },
    "//max/tests/integration/accuracy:test_pipelines_lm_eval": {
        "cpu": 2,
        "memory": 8350,
    },
    "//max/tests/integration/serve/kvcache_agent:tests": {
        "cpu": 4,
    },
    "//max/tests/integration/serve:test_sagemaker_cpu": {
        "cpu": 2,
        "memory": 1256,
    },
    "//max/tests/integration/serve:test_stop_cpu": {
        "cpu": 2,
        "memory": 1268,
    },
    "//max/tests/tests/_core_mojo:tests": {
        "cpu": 3,
        "memory": 625,
    },
    "//max/tests/tests/driver:test_device": {
        "cpu": 2,
        "memory": 262,
    },
    "//max/tests/tests/driver:test_driver": {
        "cpu": 2,
        "memory": 263,
    },
    "//max/tests/tests/driver:test_tensor": {
        "cpu": 2,
        "memory": 266,
    },
    "//max/tests/tests/entrypoints:tests": {
        "cpu": 2,
        "memory": 754,
    },
    "//max/tests/tests/graph:multi_version_tests": {
        "memory": 360,
    },
    "//max/tests/tests/graph:multi_version_tests_3.10": {
        "cpu": 2,
        "memory": 344,
    },
    "//max/tests/tests/graph:multi_version_tests_3.11": {
        "cpu": 3,
        "memory": 331,
    },
    "//max/tests/tests/graph:multi_version_tests_3.13": {
        "cpu": 3,
        "memory": 341,
    },
    "//max/tests/tests/graph:multi_version_tests_3.14": {
        "cpu": 2,
        "memory": 391,
    },
    "//max/tests/tests/graph:ops/elementwise/test_atanh": {
        "cpu": 2,
        "memory": 293,
    },
    "//max/tests/tests/graph:ops/elementwise/test_div": {
        "memory": 4972,
    },
    "//max/tests/tests/graph:ops/elementwise/test_gelu": {
        "cpu": 2,
        "memory": 341,
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_inf": {
        "cpu": 2,
        "memory": 292,
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_nan": {
        "cpu": 2,
        "memory": 296,
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_binary_ops": {
        "cpu": 2,
        "memory": 306,
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_not": {
        "cpu": 2,
        "memory": 295,
    },
    "//max/tests/tests/graph:ops/elementwise/test_sub": {
        "cpu": 2,
        "memory": 346,
    },
    "//max/tests/tests/graph:ops/reduction/test_argminmax": {
        "cpu": 2,
        "memory": 295,
    },
    "//max/tests/tests/graph:ops/test_allgather": {
        "cpu": 2,
        "memory": 370,
    },
    "//max/tests/tests/graph:ops/test_allreduce": {
        "cpu": 2,
        "memory": 301,
    },
    "//max/tests/tests/graph:ops/test_argsort": {
        "cpu": 2,
        "memory": 554,
    },
    "//max/tests/tests/graph:ops/test_band_part": {
        "cpu": 2,
        "memory": 301,
    },
    "//max/tests/tests/graph:ops/test_broadcast_to": {
        "cpu": 2,
        "memory": 317,
    },
    "//max/tests/tests/graph:ops/test_buffer": {
        "memory": 1676,
    },
    "//max/tests/tests/graph:ops/test_call": {
        "cpu": 2,
        "memory": 412,
    },
    "//max/tests/tests/graph:ops/test_cast": {
        "cpu": 2,
        "memory": 295,
    },
    "//max/tests/tests/graph:ops/test_chunk": {
        "cpu": 2,
        "memory": 383,
    },
    "//max/tests/tests/graph:ops/test_complex": {
        "memory": 302,
    },
    "//max/tests/tests/graph:ops/test_concat": {
        "memory": 308,
    },
    "//max/tests/tests/graph:ops/test_conditional": {
        "memory": 319,
    },
    "//max/tests/tests/graph:ops/test_constant": {
        "memory": 511,
    },
    "//max/tests/tests/graph:ops/test_conv": {
        "memory": 308,
    },
    "//max/tests/tests/graph:ops/test_conv3d": {
        "cpu": 2,
        "memory": 325,
    },
    "//max/tests/tests/graph:ops/test_conv_transpose": {
        "memory": 347,
    },
    "//max/tests/tests/graph:ops/test_cumsum": {
        "cpu": 2,
        "memory": 325,
    },
    "//max/tests/tests/graph:ops/test_custom": {
        "memory": 997,
    },
    "//max/tests/tests/graph:ops/test_device_chains_collectives": {
        "cpu": 2,
        "memory": 289,
    },
    "//max/tests/tests/graph:ops/test_flatten": {
        "memory": 317,
    },
    "//max/tests/tests/graph:ops/test_fold": {
        "cpu": 2,
        "memory": 574,
    },
    "//max/tests/tests/graph:ops/test_gather": {
        "memory": 338,
    },
    "//max/tests/tests/graph:ops/test_hann_window": {
        "memory": 292,
    },
    "//max/tests/tests/graph:ops/test_irfft": {
        "cpu": 2,
        "memory": 828,
    },
    "//max/tests/tests/graph:ops/test_layer_norm": {
        "memory": 300,
    },
    "//max/tests/tests/graph:ops/test_linalg": {
        "cpu": 2,
        "memory": 454,
    },
    "//max/tests/tests/graph:ops/test_min_max_overloads": {
        "cpu": 2,
        "memory": 302,
    },
    "//max/tests/tests/graph:ops/test_nonzero": {
        "cpu": 2,
        "memory": 325,
    },
    "//max/tests/tests/graph:ops/test_outer": {
        "cpu": 2,
        "memory": 354,
    },
    "//max/tests/tests/graph:ops/test_pad": {
        "memory": 295,
    },
    "//max/tests/tests/graph:ops/test_permute": {
        "cpu": 2,
        "memory": 305,
    },
    "//max/tests/tests/graph:ops/test_quantized": {
        "cpu": 2,
        "memory": 557,
    },
    "//max/tests/tests/graph:ops/test_random": {
        "cpu": 2,
        "memory": 386,
    },
    "//max/tests/tests/graph:ops/test_range": {
        "cpu": 2,
        "memory": 438,
    },
    "//max/tests/tests/graph:ops/test_rebind": {
        "memory": 308,
    },
    "//max/tests/tests/graph:ops/test_reduction": {
        "cpu": 2,
        "memory": 342,
    },
    "//max/tests/tests/graph:ops/test_repeat_interleave": {
        "memory": 581,
    },
    "//max/tests/tests/graph:ops/test_reshape": {
        "memory": 349,
    },
    "//max/tests/tests/graph:ops/test_resize": {
        "memory": 293,
    },
    "//max/tests/tests/graph:ops/test_scatter": {
        "cpu": 2,
        "memory": 342,
    },
    "//max/tests/tests/graph:ops/test_shape_to_tensor": {
        "cpu": 2,
        "memory": 313,
    },
    "//max/tests/tests/graph:ops/test_slice": {
        "cpu": 2,
        "memory": 394,
    },
    "//max/tests/tests/graph:ops/test_split": {
        "memory": 307,
    },
    "//max/tests/tests/graph:ops/test_stack": {
        "cpu": 2,
        "memory": 377,
    },
    "//max/tests/tests/graph:ops/test_tile": {
        "cpu": 2,
        "memory": 320,
    },
    "//max/tests/tests/graph:ops/test_top_k": {
        "cpu": 2,
        "memory": 289,
    },
    "//max/tests/tests/graph:ops/test_transfer": {
        "cpu": 2,
        "memory": 292,
    },
    "//max/tests/tests/graph:ops/test_transpose": {
        "memory": 336,
    },
    "//max/tests/tests/graph:ops/test_where": {
        "memory": 318,
    },
    "//max/tests/tests/graph:ops/test_while_loop": {
        "cpu": 2,
        "memory": 314,
    },
    "//max/tests/tests/graph:test_debug": {
        "cpu": 2,
        "memory": 368,
    },
    "//max/tests/tests/graph:test_device_ref": {
        "cpu": 2,
        "memory": 289,
    },
    "//max/tests/tests/graph:test_dialects": {
        "memory": 287,
    },
    "//max/tests/tests/graph:test_dtype_promotion": {
        "cpu": 2,
        "memory": 412,
    },
    "//max/tests/tests/graph:test_graph_value": {
        "cpu": 2,
        "memory": 361,
    },
    "//max/tests/tests/graph:test_non_contiguous_tensors": {
        "memory": 1273,
    },
    "//max/tests/tests/graph:test_shapes": {
        "cpu": 2,
        "memory": 289,
    },
    "//max/tests/tests/graph:test_sharding_strategy": {
        "cpu": 2,
        "memory": 316,
    },
    "//max/tests/tests/graph:test_squeeze": {
        "cpu": 2,
        "memory": 331,
    },
    "//max/tests/tests/graph:test_tensor_value": {
        "cpu": 2,
        "memory": 400,
    },
    "//max/tests/tests/graph:test_type": {
        "memory": 505,
    },
    "//max/tests/tests/graph:test_type_no_context": {
        "memory": 293,
    },
    "//max/tests/tests/graph:test_weight": {
        "memory": 302,
    },
    "//max/tests/tests/graph:utils/test_load_gguf": {
        "cpu": 2,
        "memory": 294,
    },
    "//max/tests/tests/graph:utils/test_load_safetensors": {
        "memory": 304,
    },
    "//max/tests/tests/kv_cache:test_attention": {
        "cpu": 4,
    },
    "//max/tests/tests/kv_cache:test_fp4_matmul": {
        "cpu": 2,
        "memory": 889,
    },
    "//max/tests/tests/kv_cache:test_fp8_matmul": {
        "cpu": 2,
        "memory": 746,
    },
    "//max/tests/tests/mojo-importer:mojo-importer": {
        "cpu": 3,
        "memory": 642,
    },
    "//max/tests/tests/nn:test_conv": {
        "cpu": 3,
    },
    "//max/tests/tests/nn:test_layer_norm": {
        "cpu": 5,
    },
    "//max/tests/tests/nn:test_linear": {
        "cpu": 2,
    },
    "//max/tests/tests/nn:test_module": {
        "memory": 1872,
    },
    "//max/tests/tests/nn:test_rms_norm": {
        "cpu": 2,
        "memory": 1642,
    },
    "//max/tests/tests/nn:test_sampling": {
        "cpu": 3,
        "memory": 6220,
    },
    "//max/tests/tests/nn:test_state_dict": {
        "cpu": 3,
    },
    "//max/tests/tests/nn:test_tensor_parallel_linear": {
        "cpu": 4,
    },
    "//max/tests/tests/pipelines/internvl:test_embedding_merge": {
        "cpu": 2,
        "memory": 721,
    },
    "//max/tests/tests/pipelines/internvl:test_embeddings": {
        "cpu": 2,
        "memory": 719,
    },
    "//max/tests/tests/pipelines/lib:test_audio_generation_config": {
        "memory": 719,
    },
    "//max/tests/tests/pipelines/lib:test_max_config_basic": {
        "cpu": 3,
        "memory": 704,
    },
    "//max/tests/tests/pipelines/lib:test_max_config_inheritance": {
        "cpu": 2,
        "memory": 766,
    },
    "//max/tests/tests/pipelines:test_internvl_weight_adapters": {
        "cpu": 2,
        "memory": 895,
    },
    "//max/tests/tests/pipelines:test_parse_float8_config": {
        "cpu": 2,
        "memory": 895,
    },
    "//max/tests/tests/profiler:tests": {
        "cpu": 5,
    },
    "//max/tests/tests/serve/recordreplay:test_replay": {
        "cpu": 5,
        "memory": 144,
    },
    "//max/tests/tests/serve/recordreplay:test_replay_estimation": {
        "cpu": 6,
    },
    "//max/tests/tests/serve/scheduler:test_di": {
        "cpu": 2,
        "memory": 2719,
    },
    "//max/tests/tests/serve/scheduler:test_paged_scheduler": {
        "memory": 4590,
    },
    "//max/tests/tests/serve/scheduler:test_queues": {
        "memory": 701,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler": {
        "cpu": 2,
        "memory": 666,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_config": {
        "cpu": 2,
        "memory": 698,
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_metrics": {
        "cpu": 2,
        "memory": 703,
    },
    "//max/tests/tests/serve/scheduler:test_text_batch_constructor": {
        "cpu": 2,
        "memory": 697,
    },
    "//max/tests/tests/serve/scheduler:test_token_budget": {
        "cpu": 2,
        "memory": 730,
    },
    "//max/tests/tests/serve/scheduler:test_tts_scheduler": {
        "cpu": 2,
        "memory": 2050,
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline": {
        "cpu": 2,
        "memory": 886,
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline_sampling_params": {
        "cpu": 2,
        "memory": 905,
    },
    "//max/tests/tests/serve:pipelines/test_stop_detection": {
        "cpu": 2,
        "memory": 918,
    },
    "//max/tests/tests/serve:test_async_queue": {
        "memory": 899,
    },
    "//max/tests/tests/serve:test_file_uri": {
        "cpu": 2,
        "memory": 868,
    },
    "//max/tests/tests/serve:test_kserve_routes": {
        "cpu": 2,
        "memory": 912,
    },
    "//max/tests/tests/serve:test_llm": {
        "memory": 1193,
    },
    "//max/tests/tests/serve:test_lora_integration": {
        "cpu": 2,
        "memory": 922,
    },
    "//max/tests/tests/serve:test_metrics": {
        "cpu": 2,
        "memory": 908,
    },
    "//max/tests/tests/serve:test_multiprocessing": {
        "cpu": 2,
        "memory": 913,
    },
    "//max/tests/tests/serve:test_openai_request": {
        "cpu": 2,
        "memory": 920,
    },
    "//max/tests/tests/serve:test_openai_routes": {
        "cpu": 2,
        "memory": 1268,
    },
    "//max/tests/tests/serve:test_openai_stream": {
        "cpu": 2,
        "memory": 1184,
    },
    "//max/tests/tests/serve:test_reset_prefix_cache": {
        "memory": 1222,
    },
    "//max/tests/tests/serve:test_routes": {
        "cpu": 2,
        "memory": 1175,
    },
    "//max/tests/tests/serve:test_socket_settings": {
        "cpu": 2,
        "memory": 924,
    },
    "//max/tests/tests/serve:test_stopwatch": {
        "cpu": 2,
        "memory": 910,
    },
    "//max/tests/tests/serve:test_telemetry_worker": {
        "memory": 896,
    },
    "//max/tests/tests/support:tests": {
        "memory": 481,
    },
    "//max/tests/tests/torch:tests": {
        "cpu": 2,
        "memory": 8038,
    },
    "//max/tests/tests:test_generated_dialects": {
        "cpu": 8,
        "memory": 2237,
    },
    "//max/tests/tests:test_passes": {
        "cpu": 8,
        "memory": 3119,
    },
    "//max/tests/tests:test_realization_context": {
        "cpu": 8,
        "memory": 21282,
    },
    "//max/tests/tests:test_support": {
        "cpu": 8,
        "memory": 2274,
    },
    "//max/tests/tests:test_tensor": {
        "cpu": 7,
        "memory": 21330,
    },
}
