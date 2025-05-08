import cv2

def extract_last_frame(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return False
    
    # Get total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("Error: Video has no frames.")
        return False
    
    # Set position to the last frame
    video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    # Read the last frame
    ret, frame = video.read()
    
    if not ret:
        print("Error: Could not read the last frame.")
        return False
    
    # Save the frame as an image
    cv2.imwrite(output_path, frame)
    
    # Release the video capture object
    video.release()
    
    print(f"Successfully saved last frame to {output_path}")
    return True

# Example usage
if __name__ == "__main__":
    video_path = "/workspace/video0.mp4"  # Replace with your video file path
    output_path = "last_frame.jpg"  # Replace with desired output path
    extract_last_frame(video_path, output_path)