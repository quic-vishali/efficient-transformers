# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import time
from typing import Any, Dict, List, Optional, Union
import copy

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from QEfficient import QEffFluxPipeline
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils._utils import load_json
from tests.diffusers.diffusers_utils import (
    DiffusersTestUtils,
    MADValidator,
    release_pipeline_qpc_sessions,
    release_qpc_session,
)

# Test Configuration for 256x256 resolution with 2 layers # update mad tolerance
CONFIG_PATH = "tests/diffusers/flux_test_config.json"
INITIAL_TEST_CONFIG = load_json(CONFIG_PATH)
TEST_SEED = 42
model_id = "black-forest-labs/FLUX.1-schnell"

def flux_pipeline_call_with_mad_validation(
    pipeline,
    pytorch_pipeline,
    height: int = 256,
    width: int = 256,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    export_compile_only: bool = False,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    custom_config_path: Optional[str] = None,
    use_onnx_subfunctions: bool = False,
    parallel_compile: bool = False,
    cache_threshold: Optional[float] = None,
    mad_tolerances: Dict[str, float] = None,
):
    """
    Pipeline call function that replicates the exact flow of pipeline_flux.py.__call__()
    while adding comprehensive MAD validation at each step.

    This function follows the EXACT same structure as QEffFluxPipeline.__call__()
    but adds MAD validation hooks throughout the process.
    """
    # Initialize MAD validator
    mad_validator = MADValidator(tolerances=mad_tolerances)

    device = "cpu"

    # Step 1: Load configuration, compile models
    _t_compile_start = time.time()
    pipeline.compile(
        compile_config=custom_config_path,
        parallel=parallel_compile,
        use_onnx_subfunctions=use_onnx_subfunctions,
        height=height,
        width=width,
    )
    print(f"\n[Timing] Export + Compile: {(time.time() - _t_compile_start) / 60:.2f}min")

    if export_compile_only:
        return

    # Validate all inputs
    pipeline.model.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    # Set pipeline attributes
    pipeline._guidance_scale = guidance_scale
    pipeline._interrupt = False
    batch_size = INITIAL_TEST_CONFIG["modules"]["transformer"]["specializations"]["batch_size"]

    # Step 3: Encode prompts with both text encoders
    # Use pipeline's encode_prompt method
    _t_encode_start = time.time()
    (t5_qaic_prompt_embeds, clip_qaic_pooled_prompt_embeds, text_ids, text_encoder_perf) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    (t5_torch_prompt_embeds, clip_torch_pooled_prompt_embeds, text_ids) = pytorch_pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    # Deactivate text encoder qpc sessions
    release_qpc_session(pipeline.text_encoder)
    release_qpc_session(pipeline.text_encoder_2)
    print(f"\n[Timing] Text encoding (QAIC + PyTorch ref): {(time.time() - _t_encode_start) / 60:.2f}min")

    # MAD Validation for Text Encoders
    print(" Performing MAD validation for text encoders...")
    print(f"[TEST][T5   QAIC ] mean={t5_qaic_prompt_embeds.float().mean():.6f} shape={t5_qaic_prompt_embeds.shape}")
    print(f"[TEST][T5   TORCH] mean={t5_torch_prompt_embeds.float().mean():.6f} shape={t5_torch_prompt_embeds.shape}")
    print(f"[TEST][CLIP QAIC ] mean={clip_qaic_pooled_prompt_embeds.float().mean():.6f}")
    print(f"[TEST][CLIP TORCH] mean={clip_torch_pooled_prompt_embeds.float().mean():.6f}")
    mad_validator.validate_module_mad(
        clip_qaic_pooled_prompt_embeds, clip_torch_pooled_prompt_embeds, module_name="clip_text_encoder"
    )
    mad_validator.validate_module_mad(t5_torch_prompt_embeds, t5_qaic_prompt_embeds, "t5_text_encoder")

    # Step 4: Prepare timesteps for denoising
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, device, timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # Step 5: Prepare initial latents
    num_channels_latents = pipeline.transformer.model.config.in_channels // 4
    latents, latent_image_ids = pipeline.model.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        t5_qaic_prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    print(f"[TEST][LATENTS INIT] mean={latents.float().mean():.6f} shape={latents.shape}")
    if pipeline.transformer.qpc_session is None:
        pipeline.transformer.qpc_session = QAICInferenceSession(
            str(pipeline.transformer.qpc_path), device_ids=pipeline.transformer.device_ids
        )

    # Calculate compressed latent dimension (cl) for transformer buffer allocation
    from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension

    cl, _, _ = calculate_compressed_latent_dimension(height, width, pipeline.model.vae_scale_factor)

    # Allocate output buffer for transformer
    output_buffer = {
        "output": np.zeros((batch_size, cl, pipeline.transformer.model.config.in_channels), dtype=np.float32),
    }
    pipeline.transformer.qpc_session.set_buffers(output_buffer)
    if getattr(pipeline, "enable_first_block_cache", False):
        pipeline.transformer.qpc_session.skip_buffers(
            [
                tensor_name
                for tensor_name in (
                    pipeline.transformer.qpc_session.input_names + pipeline.transformer.qpc_session.output_names
                )
                if tensor_name.startswith("prev_") or tensor_name.endswith("_RetainedState")
            ]
        )

    transformer_perf = []
    pipeline.scheduler.set_begin_index(0)

    # Step 7: Denoising loop
    _t_denoise_start = time.time()
    with pipeline.model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline._interrupt:
                continue

            # Prepare timestep embedding
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            temb = pipeline.transformer.model.time_text_embed(timestep, clip_qaic_pooled_prompt_embeds)

            # Compute AdaLN embeddings for dual transformer blocks
            adaln_emb = []
            for block_idx in range(len(pipeline.transformer.model.transformer_blocks)):
                block = pipeline.transformer.model.transformer_blocks[block_idx]
                f1 = block.norm1.linear(block.norm1.silu(temb)).chunk(6, dim=1)
                f2 = block.norm1_context.linear(block.norm1_context.silu(temb)).chunk(6, dim=1)
                adaln_emb.append(torch.cat(list(f1) + list(f2)))
            adaln_dual_emb = torch.stack(adaln_emb)

            # Compute AdaLN embeddings for single transformer blocks
            adaln_emb = []
            for block_idx in range(len(pipeline.transformer.model.single_transformer_blocks)):
                block = pipeline.transformer.model.single_transformer_blocks[block_idx]
                f1 = block.norm.linear(block.norm.silu(temb)).chunk(3, dim=1)
                adaln_emb.append(torch.cat(list(f1)))
            adaln_single_emb = torch.stack(adaln_emb)

            # Compute output AdaLN embedding
            temp = pipeline.transformer.model.norm_out
            adaln_out = temp.linear(temp.silu(temb))

            # Normalize timestep to [0, 1] range
            timestep = timestep / 1000

            # Prepare all inputs for transformer inference
            inputs_aic = {
                "hidden_states": latents.detach().numpy(),
                "encoder_hidden_states": t5_qaic_prompt_embeds.detach().numpy(),
                "pooled_projections": clip_qaic_pooled_prompt_embeds.detach().numpy(),
                "timestep": timestep.detach().numpy(),
                "img_ids": latent_image_ids.detach().numpy(),
                "txt_ids": text_ids.detach().numpy(),
                "adaln_emb": adaln_dual_emb.detach().numpy(),
                "adaln_single_emb": adaln_single_emb.detach().numpy(),
                "adaln_out": adaln_out.detach().numpy(),
            }
            if getattr(pipeline, "enable_first_block_cache", False):
                stage_cache_threshold = 0.0 if cache_threshold is None else cache_threshold
                inputs_aic["cache_threshold"] = np.array(stage_cache_threshold, dtype=np.float32)

            # MAD Validation for Transformer - PyTorch reference inference
            noise_pred_torch = pytorch_pipeline.transformer(
                hidden_states=latents,
                encoder_hidden_states=t5_torch_prompt_embeds,
                pooled_projections=clip_torch_pooled_prompt_embeds,
                timestep=torch.tensor(timestep),
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                return_dict=False,
            )[0]

            # Run transformer inference and measure time
            start_transformer_step_time = time.time()
            outputs = pipeline.transformer.qpc_session.run(inputs_aic)
            end_transformer_step_time = time.time()
            transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

            noise_pred = torch.from_numpy(outputs["output"])

            # Transformer MAD validation
            print(f"[TEST][TRANSFORMER IN  step {i}] latents mean={latents.float().mean():.6f}")
            print(f"[TEST][TRANSFORMER OUT step {i}] qaic  mean={outputs['output'].mean():.6f}")
            print(f"[TEST][TRANSFORMER OUT step {i}] torch mean={noise_pred_torch.float().mean():.6f}")
            mad_validator.validate_module_mad(
                noise_pred_torch.detach().cpu().numpy(),
                outputs["output"],
                "transformer",
                f"step {i} (t={t.item():.1f})",
            )

            # Update latents using scheduler
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Handle dtype mismatch
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            # Update progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    release_qpc_session(pipeline.transformer)
    print(f"\n[Timing] Denoising loop: {(time.time() - _t_denoise_start) / 60:.2f}min")
    # Step 8: Decode latents to images
    if output_type == "latent":
        image = latents
        vae_decode_perf = 0.0  # No VAE decoding for latent output
    else:
        # Unpack and denormalize latents
        latents = pipeline.model._unpack_latents(latents, height, width, pipeline.model.vae_scale_factor)

        # Denormalize latents
        latents = (latents / pipeline.vae_decode.model.scaling_factor) + pipeline.vae_decode.model.shift_factor
        # Initialize VAE decoder inference session
        if pipeline.vae_decode.qpc_session is None:
            pipeline.vae_decode.qpc_session = QAICInferenceSession(
                str(pipeline.vae_decode.qpc_path), device_ids=pipeline.vae_decode.device_ids
            )

        # Allocate output buffer for VAE decoder
        output_buffer = {"sample": np.zeros((batch_size, 3, height, width), dtype=np.float32)}
        pipeline.vae_decode.qpc_session.set_buffers(output_buffer)

        # MAD Validation for VAE
        # PyTorch reference inference
        image_torch = pytorch_pipeline.vae.decode(latents, return_dict=False)[0]

        # Run VAE decoder inference and measure time
        inputs = {"latent_sample": latents.numpy()}
        start_decode_time = time.time()
        image = pipeline.vae_decode.qpc_session.run(inputs)
        end_decode_time = time.time()
        vae_decode_perf = end_decode_time - start_decode_time
        release_qpc_session(pipeline.vae_decode)

        # # VAE MAD validation
        print(f"[TEST][VAE IN ] latents mean={latents.float().mean():.6f}")
        print("image_torch - ", image_torch.dtype)
        print('image["sample"]', image["sample"].dtype)
        print(f"[TEST][VAE OUT QAIC ] mean={image['sample'].mean():.6f}")
        print(f"[TEST][VAE OUT TORCH] mean={image_torch.float().mean():.6f}")
        mad_validator.validate_module_mad(image_torch.detach().cpu().numpy(), image["sample"], "vae_decoder")

        # Post-process image
        image_tensor = torch.from_numpy(image["sample"])
        image = pipeline.model.image_processor.postprocess(image_tensor, output_type=output_type)
        print(f"image postprocess - {image}")
        img_np = np.array(image)

        print("Shape:", img_np.shape)
        print("Dtype:", img_np.dtype)
        print("Min:", img_np.min())
        print("Max:", img_np.max())

        print("\nPixel values:")
        print(img_np)

    # Build performance metrics
    perf_metrics = [
        ModulePerf(module_name="text_encoder", perf=text_encoder_perf[0]),
        ModulePerf(module_name="text_encoder_2", perf=text_encoder_perf[1]),
        ModulePerf(module_name="transformer", perf=transformer_perf),
        ModulePerf(module_name="vae_decoder", perf=vae_decode_perf),
    ]

    print(
        f"\n[Timing] Phase summary:"
        f"\n  Export + Compile : (see above)"
        f"\n  Text encoding    : (see above)"
        f"\n  Denoising loop   : (see above)"
        f"\n  CLIP encoder     : {text_encoder_perf[0] / 60:.4f}min"
        f"\n  T5 encoder       : {text_encoder_perf[1] / 60:.4f}min"
        f"\n  Transformer steps: {sum(transformer_perf) / 60:.4f}min total ({len(transformer_perf)} steps)"
        f"\n  VAE decode       : {vae_decode_perf / 60:.4f}min"
    )

    return QEffPipelineOutput(
        pipeline_module=perf_metrics,
        images=image,
    )


def _build_flux_pipeline(enable_first_block_cache: bool = False):
    """Build Flux test pipelines with random-initialized (dummy) weights."""
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)

    _t_load_start = time.time()
    if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
        config = INITIAL_TEST_CONFIG["model_setup"]

        # Build random-init components from model configs (no pretrained weights).
        vae_config = AutoencoderKL.load_config(model_id, subfolder="vae")
        transformer_config = FluxTransformer2DModel.load_config(model_id, subfolder="transformer")
        scheduler_cfg = FlowMatchEulerDiscreteScheduler.load_config(model_id, subfolder="scheduler")

        transformer_config["num_layers"] = config["num_transformer_layers"]
        transformer_config["num_single_layers"] = config["num_single_layers"]

        vae = AutoencoderKL.from_config(vae_config)
        transformer = FluxTransformer2DModel.from_config(transformer_config)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_cfg)

        clip_text_encoder_cfg = CLIPTextModel.config_class.from_pretrained(model_id, subfolder="text_encoder")
        t5_text_encoder_cfg = T5EncoderModel.config_class.from_pretrained(model_id, subfolder="text_encoder_2")

        # Reduce text-encoder depth for faster export/compile in this test.
        clip_text_encoder_cfg.num_hidden_layers = 1
        t5_text_encoder_cfg.num_layers = 1

        text_encoder = CLIPTextModel(clip_text_encoder_cfg)
        text_encoder_2 = T5EncoderModel(t5_text_encoder_cfg)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")

        pytorch_pipeline = FluxPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        vae.eval()
        transformer.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        pipeline = QEffFluxPipeline(
            copy.deepcopy(pytorch_pipeline),
            enable_first_block_cache=enable_first_block_cache,
        )
    else:
        pytorch_pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

        pytorch_pipeline.vae.eval()
        pytorch_pipeline.transformer.eval()
        if pytorch_pipeline.text_encoder is not None:
            pytorch_pipeline.text_encoder.eval()
        if pytorch_pipeline.text_encoder_2 is not None:
            pytorch_pipeline.text_encoder_2.eval()

        pipeline = QEffFluxPipeline(
            copy.deepcopy(pytorch_pipeline),
            pretrained_model_name_or_path=model_id,
            enable_first_block_cache=enable_first_block_cache,
        )

    print(f"\n[Timing] Model loading: {(time.time() - _t_load_start) / 60:.2f}min")
    return pipeline, pytorch_pipeline


@pytest.fixture(scope="function")
def flux_pipeline():
    return _build_flux_pipeline(enable_first_block_cache=False)


@pytest.fixture(scope="function")
def flux_pipeline_first_block_cache():
    return _build_flux_pipeline(enable_first_block_cache=True)


def _run_flux_pipeline_test_case(
    flux_pipeline_data,
    config,
    test_label: str,
    export_compile_only: bool = False,
    pipeline_call_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Comprehensive Flux pipeline test that follows the exact same flow as pipeline_flux.py:
    - 256x256 resolution - 2 transformer layers
    - MAD validation
    - Functional image generation test
    - Export/compilation checks
    - Returns QEffPipelineOutput with performance metrics
    """
    pipeline, pytorch_pipeline = flux_pipeline_data

    # Print test header
    DiffusersTestUtils.print_test_header(
        test_label,
    )

    # Test parameters
    test_prompt = config["pipeline_params"]["test_prompt"]
    num_inference_steps = config["pipeline_params"]["num_inference_steps"]
    guidance_scale = config["pipeline_params"]["guidance_scale"]
    max_sequence_length = config["pipeline_params"]["max_sequence_length"]

    # Generate with MAD validation
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    start_time = time.time()

    try:
        pipeline_call_overrides = pipeline_call_overrides or {}

        # Run the pipeline with integrated MAD validation (follows exact pipeline flow)
        result = flux_pipeline_call_with_mad_validation(
            pipeline,
            pytorch_pipeline,
            height=config["model_setup"]["height"],
            width=config["model_setup"]["width"],
            prompt=test_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            custom_config_path=CONFIG_PATH,
            generator=generator,
            mad_tolerances=config["mad_validation"]["tolerances"],
            use_onnx_subfunctions=config["pipeline_params"]["use_onnx_subfunctions"],
            parallel_compile=True,
            export_compile_only=export_compile_only,
            **pipeline_call_overrides,
        )

        execution_time = time.time() - start_time

        if config["validation_checks"]["onnx_export"]:
            # Check if ONNX files exist (basic check)
            print("\n ONNX Export Validation:")
            for module_name in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]:
                module_obj = getattr(pipeline, module_name, None)
                if module_obj and hasattr(module_obj, "onnx_path") and module_obj.onnx_path:
                    DiffusersTestUtils.check_file_exists(str(module_obj.onnx_path), f"{module_name} ONNX")

        if config["validation_checks"]["compilation"]:
            # Check if QPC files exist (basic check)
            print("\n Compilation Validation:")
            for module_name in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]:
                module_obj = getattr(pipeline, module_name, None)
                if module_obj and hasattr(module_obj, "qpc_path") and module_obj.qpc_path:
                    DiffusersTestUtils.check_file_exists(str(module_obj.qpc_path), f"{module_name} QPC")

        DiffusersTestUtils.print_artifact_sizes({
            **{
                f"{mn} ONNX": str(getattr(pipeline, mn).onnx_path)
                for mn in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]
                if getattr(pipeline, mn, None) and getattr(pipeline, mn).onnx_path
            },
            **{
                f"{mn} QPC": str(getattr(pipeline, mn).qpc_path)
                for mn in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]
                if getattr(pipeline, mn, None) and getattr(pipeline, mn).qpc_path
            },
        })

        if export_compile_only:
            return

      # Validate image generation
        if config["pipeline_params"]["validate_gen_img"]:
            assert result is not None, "Pipeline returned None"
            assert hasattr(result, "images"), "Result missing 'images' attribute"
            assert len(result.images) > 0, "No images generated"

            generated_image = result.images[0]
            expected_size = (config["model_setup"]["height"], config["model_setup"]["width"])
            # Validate image properties using utilities
            image_validation = DiffusersTestUtils.validate_image_generation(
                generated_image, expected_size, config["pipeline_params"]["min_image_variance"]
            )

            print("\n IMAGE VALIDATION PASSED")
            print(f"   - Size: {image_validation['size']}")
            print(f"   - Mode: {image_validation['mode']}")
            print(f"   - Variance: {image_validation['variance']:.2f}")
            print(f"   - Mean pixel value: {image_validation['mean_pixel_value']:.2f}")
            file_path = "test_flux_64x64_2layers.png"
            # Save test image
            generated_image.save(file_path)

            if os.path.exists(file_path):
                print(f"Image saved successfully at: {file_path}")
            else:
                print("Image was not saved.")

        # Print test summary using utilities
        print(f"\nTotal execution time: {execution_time:.4f}s")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise
    finally:
        release_pipeline_qpc_sessions(pipeline, ["text_encoder", "text_encoder_2", "transformer", "vae_decode"])


@pytest.mark.flux
@pytest.mark.diffusion_models
@pytest.mark.non_qaic
def test_export_compile(flux_pipeline):
    _run_flux_pipeline_test_case(
        flux_pipeline,
        INITIAL_TEST_CONFIG,
        test_label="Test Export Compile",
        export_compile_only=True,
    )

@pytest.mark.flux
@pytest.mark.diffusion_models
@pytest.mark.qaic
def test_generate(flux_pipeline):
    _run_flux_pipeline_test_case(
        flux_pipeline,
        INITIAL_TEST_CONFIG,
        test_label="Test Generate"
    )


@pytest.mark.flux
@pytest.mark.diffusion_models
@pytest.mark.non_qaic
def test_export_compile_first_block_cache(flux_pipeline_first_block_cache):
    _run_flux_pipeline_test_case(
        flux_pipeline_first_block_cache,
        INITIAL_TEST_CONFIG,
        test_label="Test Export Compile First Block Cache",
        export_compile_only=True,
    )


@pytest.mark.flux
@pytest.mark.diffusion_models
@pytest.mark.qaic
def test_generate_first_block_cache(flux_pipeline_first_block_cache):
    _run_flux_pipeline_test_case(
        flux_pipeline_first_block_cache,
        INITIAL_TEST_CONFIG,
        test_label="Test Generate First Block Cache",
        pipeline_call_overrides={"cache_threshold": 0.0},
    )


if __name__ == "__main__":
    # This allows running the test file directly for debugging
    pytest.main([__file__, "-v", "-s", "-m", "flux"])
# pytest tests/diffusers/test_flux.py -m flux -v -s --tb=short
