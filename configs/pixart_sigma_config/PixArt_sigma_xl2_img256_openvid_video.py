_base_ = ['../PixArt_xl2_internal.py']
data_root = 'dataset/data/train/OpenVidHD.csv'
root = "dataset/video"
#image_list_json = ['data_info.json']

data = dict(
    type='InternalDataMSSigma', root='InternData', transform='default_train',
    load_vae_feat=False, load_t5_feat=False,
)
#image_size = 512
image_size = 256
num_frames = 16
frame_interval = 3

# model setting
model = 'STDiT-XL/2'
mixed_precision = 'fp16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = True
load_from = "output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
resume_from = None
vae_pretrained = "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 0.5

# training setting
num_workers = 2
train_batch_size = 36  # 48 as default
num_epochs = 200  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
#optimizer = dict(type='AdamWWrapper', lr=2e-5, weight_decay=0.03, betas=(0.9, 0.999))
lr_schedule_args = dict(num_warmup_steps=1000)
image_dropout_prob = 0.1
freeze = "not_temporal_and_xembedder"

eval_sampling_steps = 250
visualize = True
log_interval = 20
save_model_epochs = 5
save_model_steps = 2500
work_dir = 'output/debug'

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 0.5
model_max_length = 300
class_dropout_prob = 0.1

# Inference setting
noise_scheduler_kwargs=dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="linear", steps_offset=1, clip_sample=false)
