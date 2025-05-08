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
from models.utils import (_crop_and_resize, _crop_and_resize_pad,
                          _scale_height_and_pad, write_mp4)
from PIL import Image
from transformers import CLIPVisionModel

prompt = "This is an TikTok style influencer video. the woman upper body is visible as she is standing behind the counter. She picks up the blue pink purple Yerba Magic Product bag in her hands. She is talking excitedly and happy. she points and gestures. facing the camera" 
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

refer_images = [ '/workspace/finetrainers/A2/assets/woman.jpeg', '/workspace/finetrainers/A2/assets/object.jpeg', '/workspace/finetrainers/A2/assets/background.jpeg'] 
last_frame = "/workspace/finetrainers/A2/last_frame.jpg"
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

# download models
#snapshot_download(repo_id="Skywork/SkyReels-A2", local_dir="Skywork/SkyReels-A2")

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
    if image_id == 0: 
        image_vae = _crop_and_resize_pad(image, height=height, width=width) # ref image
    elif image_id == 1: 
        image_vae = _scale_height_and_pad(image, height=height, width=width) # object image
        # Save the scaled object image
        scaled_image_path = os.path.join(os.path.dirname(video_path), f"scaled_object.png")
        image_vae.save(scaled_image_path)
        print(f"Saved scaled object image to {scaled_image_path}")
    else:
        image_vae = _crop_and_resize(image, height=height, width=width) # background image
    
    image_vae = video_processor.preprocess(image_vae, height=height, width=width).to(memory_format=torch.contiguous_format) # (1, 3, 480, 320)
    image_vae = image_vae.unsqueeze(2).to(device, dtype=torch.float32)
    vae_image_list.append(image_vae) #.to(device, dtype=dtype))

# forward
generator = torch.Generator(device).manual_seed(seed) 
video_pt = pipe(
    image_clip=clip_image_list, 
    image_vae=vae_image_list,
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    first_frame = last_frame,
    height=480, 
    width=width, 
    num_frames=81, 
    guidance_scale=5.0,
    generator=generator,
    output_type="pt",
    num_inference_steps=30,
    vae_combine="before",
    tea_cache_l1_thresh=tea_cache_l1_thresh,
    tea_cache_model_id=tea_cache_model_id,
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

write_mp4(video_path, final_images, fps=16) 



# Simply convert each frame to numpy array without creating side-by-side comparison
final_images = []
for frame in video_generate:
    # Convert PIL Image to numpy array
    frame_np = np.array(frame)
    final_images.append(frame_np)
# Write the video directly with just the generated frames
write_mp4("simple.mp4", final_images, fps=16)