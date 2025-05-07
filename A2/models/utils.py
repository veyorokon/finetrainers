import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image


def _crop_and_resize_pad(image, height=480, width=720):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        pad = int((((height / width) * image_width) - image_height) / 2.)
        padded_image = np.ones((image_height + pad * 2, image_width, 3), dtype=np.uint8) * 255
        # padded_image = np.zeros((image_height + pad * 2, image_width, 3), dtype=np.uint8)
        padded_image[pad:pad+image_height, :] = image
        image = Image.fromarray(padded_image).resize((width, height))
    else:
        pad = int((((width / height) * image_height) - image_width) / 2.)
        padded_image = np.ones((image_height, image_width + pad * 2, 3), dtype=np.uint8) * 255
        # padded_image = np.zeros((image_height, image_width + pad * 2, 3), dtype=np.uint8) 
        padded_image[:, pad:pad+image_width] = image
        image = Image.fromarray(padded_image).resize((width, height))
    return image 


def _crop_and_resize(image, height=512, width=512):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width/width*height)
        top = (image_height - croped_height) // 2
        image = image[top:top+croped_height, :]
        image = Image.fromarray(image).resize((width, height))

    return image


def _scale_height_and_pad(image, height=480, width=720, scale_factor=0.5, pad_value=255):
    """
    Scales the image to a percentage of its height and resizes to specified dimensions.
    The remaining space is filled with padding.
    
    Args:
        image: PIL Image or numpy array
        height: Target height of the output image
        width: Target width of the output image
        scale_factor: Float between 0 and 1, percentage of height to scale to
        pad_value: Value to use for padding (255 for white, 0 for black)
        
    Returns:
        PIL Image with scaled content and fixed dimensions
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        
    # First resize to target dimensions while preserving aspect ratio
    image_height, image_width, channels = image.shape
    if image_height / image_width < height / width:
        pad = int((((height / width) * image_width) - image_height) / 2.)
        padded_image = np.ones((image_height + pad * 2, image_width, channels), dtype=np.uint8) * pad_value
        padded_image[pad:pad+image_height, :] = image
        image = Image.fromarray(padded_image).resize((width, height))
    else:
        pad = int((((width / height) * image_height) - image_width) / 2.)
        padded_image = np.ones((image_height, image_width + pad * 2, channels), dtype=np.uint8) * pad_value
        padded_image[:, pad:pad+image_width] = image
        image = Image.fromarray(padded_image).resize((width, height))
    
    # Convert back to numpy for scaling
    image = np.array(image)
    
    # Calculate how much smaller the scaled content should be
    new_height = int(height * scale_factor)
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    
    # Create a blank canvas of the target size
    result_image = np.ones((height, width, 3), dtype=np.uint8) * pad_value
    
    # Resize the image to the scaled dimensions
    scaled_content = Image.fromarray(image).resize((new_width, new_height), Image.LANCZOS)
    scaled_content = np.array(scaled_content)
    
    # Calculate position to place scaled content (centered)
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2
    
    # Place the scaled content in the center
    result_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled_content
    
    return Image.fromarray(result_image)
    

def write_mp4(video_path, samples, fps=14, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate, 
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])
