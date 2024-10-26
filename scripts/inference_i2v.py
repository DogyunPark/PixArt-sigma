import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from mmcv.runner import LogBuffer
from PIL import Image
from einops import rearrange
from torch.utils.data import RandomSampler
import random
from ema_pytorch import EMA

from diffusion import IDDPM, DPMS, FluxPipelineI2V, FlowMatchEulerDiscreteScheduler
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint, load_checkpoint_pixart
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.utils.nn import append_dims
from diffusion.openviddata.datasets import DatasetFromCSV, get_transforms_image, get_transforms_video
from diffusion.openviddata import save_sample
from diffusion.model.respace import FlowWrappedModel
from tools.inference import load_data_prompts

from mmengine.config import Config

warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


@torch.inference_mode()
def log_validation(model, step, device, vae, text_encoder, tokenizer, val_scheduler):
    torch.cuda.empty_cache()
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.]], device=device).repeat(1, 1)
    null_y = torch.load(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
    null_y = null_y['uncond_prompt_embeds'].to(device)

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []
    latents = []
    
    validation_pipeline.transformer.eval()

    for prompt in validation_prompts:
        if validation_noise is not None:
            z = torch.clone(validation_noise).to(device)
        else:
            z = torch.randn(1, 4, config.num_frames, latent_size, latent_size, device=device)
        embed = torch.load(f'output/tmp/{prompt}_{max_length}token.pth', map_location='cpu')
        caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
        image_embs = embed['video'].to(device)[None]
        
        b1, _, t1, h1, w1 = image_embs.shape
        first_frame_cond = torch.zeros_like(image_embs)
        first_frame_cond[:,:,0] = image_embs[:,:,0].detach().clone()
        first_frame_mask = torch.zeros((b1, 1, t1, h1, w1)).to(device, torch.float16)
        first_frame_mask[:,:,0] = 1.

        first_frame_cond_null = torch.zeros_like(image_embs)
        first_frame_mask_null = torch.zeros((b1, 1, t1, h1, w1)).to(device, torch.float16)
        
        x_cond = torch.cat([first_frame_cond, first_frame_mask], dim=1)
        x_cond_null = torch.cat([first_frame_cond_null, first_frame_mask_null], dim=1)
        
        model_kwargs = dict(mask=emb_masks)

        denoised = validation_pipeline(
            height=config.image_size,
            width=config.image_size,
            num_frames=config.num_frames,
            num_inference_steps=50,
            guidance_scale=7.5,
            prompt_embeds=caption_embs,
            prompt_embeds_mask=emb_masks,
            uncond_prompt_embeds=null_y,
            image_embeds=x_cond,
            image_embeds_null=x_cond_null,
            max_sequence_length=max_length,
            device=device,
            #FlowModel=FlowModel,
        )
        latents.append(denoised)

    torch.cuda.empty_cache()
    for prompt, latent in zip(validation_prompts, latents):
        latent = latent.to(torch.float16)
        bs = 2
        B = latent.shape[0]
        x = rearrange(latent, "B C T H W -> (B T) C H W")
        x_out = []
        for i in range(0, x.shape[0], bs):
            x_bs = x[i : i + bs]
            x_bs = vae.decode(x_bs.detach() / vae.config.scaling_factor).sample
            x_out.append(x_bs)
        x = torch.cat(x_out, dim=0)
        samples = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        sample_save_pth = os.path.join(config.work_dir, 'samples_{}'.format(step))
        os.makedirs(sample_save_pth, exist_ok=True)
        save_sample(samples[0], fps=8, save_path=os.path.join(sample_save_pth, prompt))

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--promptdir', type=str, help='the dir to resume the training')
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    #config = read_config(args.config)
    config = Config.fromfile(args.config)
    config.promptdir = args.promptdir
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        #even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    validation_noise = torch.randn(1, 4, config.num_frames, latent_size, latent_size, device='cpu') if getattr(config, 'deterministic_validation', False) else None
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    vae = None
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, torch_dtype=torch.float16).to(accelerator.device)
        config.scale_factor = vae.config.scaling_factor
    tokenizer = text_encoder = None
    if not config.data.load_t5_feat:
        tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
    
    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(**config.noise_scheduler_kwargs)
    val_scheduler = FlowMatchEulerDiscreteScheduler(**config.noise_scheduler_kwargs)
    logger.info(f"vae scale factor: {config.scale_factor}")

    if config.visualize:
        # preparing embeddings for visualization. We put it here for saving GPU memory
        #validation_prompts = config.validation_prompts
        filename_list, data_list, validation_prompts = load_data_prompts(config.promptdir, video_size=(config.image_size, config.image_size), video_frames=config.num_frames)
        #import pdb; pdb.set_trace()
        skip = True
        Path('output/tmp').mkdir(parents=True, exist_ok=True)
        for prompt in validation_prompts:
            if not (os.path.exists(f'output/tmp/{prompt}_{max_length}token.pth')
                    and os.path.exists(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            if config.data.load_t5_feat and (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {args.pipeline_load_from} ...")
                tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
                text_encoder = T5EncoderModel.from_pretrained(
                    args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
            for prompt_idx, prompt in enumerate(validation_prompts):
                txt_tokens = tokenizer(
                    prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(accelerator.device)
                caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]

                image_data = data_list[prompt_idx].to(accelerator.device)
                image_data = rearrange(image_data, "C T H W -> T C H W")
                bs = 2
                x_out = []
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                        for i in range(0, image_data.shape[0], bs):
                            x_bs = image_data[i : i + bs]
                            x_bs = vae.encode(x_bs).latent_dist.sample().mul_(config.scale_factor)
                            x_out.append(x_bs)
                        x = torch.cat(x_out, dim=0)
                        x = rearrange(x, "T C H W -> C T H W")
                torch.save(
                    {'caption_embeds': caption_emb, 'emb_mask': txt_tokens.attention_mask, 'video': x.cpu()},
                    f'output/tmp/{prompt}_{max_length}token.pth')
                del txt_tokens
                del caption_emb
                del x

            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            torch.save(
                {'uncond_prompt_embeds': null_token_emb, 'uncond_prompt_embeds_mask': null_tokens.attention_mask},
                f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
            del null_tokens
            del null_token_emb
            
            if config.data.load_t5_feat:
                del tokenizer
                del text_encoder
            flush()

    model_kwargs = {"pe_interpolation": config.pe_interpolation, "config": config,
                    "model_max_length": max_length, "qk_norm": config.qk_norm,
                    "kv_compress_config": kv_compress_config, "micro_condition": config.micro_condition,
                    "weight_freeze": config.weight_freeze}

    # build models
    #train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    #FlowModel = FlowWrappedModel(config.reparameterization)
    z_latent_size = (config.num_frames, config.image_size // 8, config.image_size // 8)
    #latent_size = vae.get_latent_size(input_size)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=z_latent_size,
                        learn_sigma=False,
                        pred_sigma=False,
                        in_channels=8+1,
                        **model_kwargs).train()
    
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.load_from is not None:
        config.load_from = args.load_from
    if config.load_from is not None:
        # missing, unexpected = load_checkpoint(
        #     config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_length)
        load_checkpoint_pixart(model, config.load_from, first_layer_ignore=False, last_layer_ignore=False)

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    validation_pipeline = FluxPipelineI2V(scheduler=val_scheduler,
                                    vae=vae,
                                    text_encoder_2=text_encoder,
                                    tokenizer_2=tokenizer,
                                    transformer=model,
                                )

    model = accelerator.prepare(model)
    global_step = 'EVAL_256_3'
    log_validation(model, global_step, device=accelerator.device, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, val_scheduler=val_scheduler)
