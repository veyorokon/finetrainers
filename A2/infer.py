import os

import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video, load_image
from diffusers.video_processor import VideoProcessor
from huggingface_hub import snapshot_download
from models.pipeline_a2 import A2Pipeline
from models.transformer_a2 import A2Model
from models.utils import _crop_and_resize, _crop_and_resize_pad, write_mp4
from PIL import Image
from transformers import CLIPVisionModel

prompt = "This is an TikTok style influencer video. the woman upper body is visible as she is standing behind the counter. She picks up the blue pink purple Yerba Magic Product bag in her hands. She is talking excitedly and happy. she points and gestures. facing the camera" 
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

refer_images = [ '/workspace/finetrainers/A2/assets/woman.jpeg', '/workspace/finetrainers/A2/assets/object.jpeg', '/workspace/finetrainers/A2/assets/background.jpeg'] 
last_frame = "/workspace/finetrainers/A2/assets/last_frame.jpg"
width = 832
height = 480 
seed = 42

# model parameters 
device = "cuda"
video_path = "output.mp4"
pipeline_path = "/dev/shm/models"
dtype = torch.bfloat16
use_teacache = True 

if use_teacache:
    tea_cache_l1_thresh = 0.3
    tea_cache_model_id = "Wan2.1-I2V-14B-480P"
else:
    tea_cache_l1_thresh = None
    tea_cache_model_id = ""

# load models 
image_encoder = CLIPVisionModel.from_pretrained(pipeline_path, subfolder="image_encoder", torch_dtype=torch.float32) 
vae = AutoencoderKLWan.from_pretrained(pipeline_path, subfolder="vae", torch_dtype=torch.float32)

# print("load transformer...")
model_path = os.path.join(pipeline_path, 'transformer')
transformer = A2Model.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=True)
# # transformer.save_pretrained("transformer", max_shard_size="5GB") 
transformer.to(device, dtype=dtype) 

pipe = A2Pipeline.from_pretrained(pipeline_path, transformer=transformer, vae=vae, image_encoder=image_encoder, torch_dtype=dtype)

scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8)
pipe.scheduler = scheduler 
pipe.to(device)

VAE_SCALE_FACTOR_SPATIAL = 8
video_processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR_SPATIAL)

# prepare reference images
clip_image_list = []
vae_image_list = []
for image_id, image_path in enumerate(refer_images): 
    image = load_image(image=image_path).convert("RGB")
    # for clip 
    image_clip = _crop_and_resize_pad(image, height=512, width=512) 
    clip_image_list.append(image_clip)
    
    # for vae 
    if image_id == 0 or image_id == 1: 
        image_vae = _crop_and_resize_pad(image, height=height, width=width) # ref image
    else:
        image_vae = _crop_and_resize(image, height=height, width=width) # background image
    
    image_vae = video_processor.preprocess(image_vae, height=height, width=width).to(memory_format=torch.contiguous_format) # (1, 3, 480, 320)
    image_vae = image_vae.unsqueeze(2).to(device, dtype=torch.float32)
    vae_image_list.append(image_vae) #.to(device, dtype=dtype))

generator = torch.Generator(device).manual_seed(seed) 
# Process last frame separately (will be added to control latents)
if last_frame:
    # Load and preprocess the last frame
    last_frame_image = load_image(image=last_frame).convert("RGB")
    #last_frame_image = _crop_and_resize(last_frame_image, height=height, width=width)
    
    # Print shape of the reference images for comparison
    print(f"Shape of a reference vae image: {vae_image_list[0].shape}")
    
    last_frame_image = video_processor.preprocess(last_frame_image, height=height, width=width).to(memory_format=torch.contiguous_format)
    print(f"Shape after preprocess: {last_frame_image.shape}")
    
    # Add batch and frame dimensions
    last_frame_image = last_frame_image.unsqueeze(2)
    print(f"Shape after unsqueeze(2): {last_frame_image.shape}")
    
    # For a 4D tensor -> 5D, we need to use repeat with 5 dimensions
    print(f"Number of dimensions: {last_frame_image.dim()}")
    
    # Create mini-video with repeated frames
    if last_frame_image.dim() == 5:
        # Already has batch dimension
        last_frame_video = torch.cat([last_frame_image] * 4, dim=2)
        print(f"Used concat for 5D tensor, new shape: {last_frame_video.shape}")
    else:
        # Add batch dimension first, then concat
        last_frame_image = last_frame_image.unsqueeze(0)
        print(f"Added batch dim, shape: {last_frame_image.shape}")
        last_frame_video = torch.cat([last_frame_image] * 4, dim=2)
        print(f"Used concat for tensor, new shape: {last_frame_video.shape}")
    
    last_frame_video = last_frame_video.to(device, dtype=torch.float32)
    
    # Encode with VAE (do separately from references)
    with torch.no_grad():
        last_frame_latent = pipe.vae.encode(last_frame_video).latent_dist.sample(generator)
        
    print(f"Encoded last frame as separate latent with shape: {last_frame_latent.shape}")
else:
    last_frame_latent = None

# forward
video_pt = pipe(
    image_clip=clip_image_list, 
    image_vae=vae_image_list,
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    height=480, 
    width=width, 
    num_frames=81, 
    guidance_scale=5.0,
    generator=generator,
    output_type="pt",
    num_inference_steps=50,
    vae_combine="before",
    tea_cache_l1_thresh=tea_cache_l1_thresh,
    tea_cache_model_id=tea_cache_model_id,
    first_frame_latent=last_frame_latent,  # Add first frame latent for separate conditioning
).frames


# combine results
batch_size = video_pt.shape[0]
batch_video_frames = []
for batch_idx in range(batch_size):
    pt_image = video_pt[batch_idx]
    pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
    pt_image = pt_image[12:]
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    batch_video_frames.append(image_pil)

video_generate = batch_video_frames[0] 
final_images = []
for q in range(len(video_generate)): 
    frame1 = _crop_and_resize_pad(load_image(image=refer_images[0]), height, width) 
    frame2 = _crop_and_resize_pad(load_image(image=refer_images[1]), height, width) 
    frame3 = _crop_and_resize_pad(load_image(image=refer_images[2]), height, width) 
    frame4 = Image.fromarray(np.array(video_generate[q])).convert("RGB")
    result = Image.new('RGB', (width * 4, height),color="white")
    result.paste(frame1, (0, 0)) 
    result.paste(frame2, (width, 0)) 
    result.paste(frame3, (width*2, 0)) 
    result.paste(frame4, (width*3, 0)) 
    final_images.append(np.array(result))

write_mp4(video_path, final_images, fps=15) 