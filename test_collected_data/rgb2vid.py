import cv2
import numpy as np
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description="Convert a sequence of images to a video.")
parser.add_argument("--dir", 
    type=str, 
    default="/datasets/ATI/MDE/orbbec_realworld_dataset/comlab_scene_dim_normal_topology2/pair_006_exposure_4000_gain_32", 
    help="Path to the folder containing images."
)

if __name__ == "__main__":
    # Set the path to the folder containing the images
    args = parser.parse_args()
    directory = args.dir
    
    # Get a list of all image files in the folder
    image_files = sorted(glob(os.path.join(f"{directory}/lap_*/rgb/", "*.png")))
    
    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the specified folder.")
        exit(1)
    
    # Read the first image to get the dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    # Define the codec and create a VideoWriter object
    video_filename = "./output_vid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'MJPG', etc.
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (width, height))
    
    # Loop through each image file and write it to the video
    for image_file in image_files:
        img = cv2.imread(image_file)
        video_writer.write(img)
    
    # Release the VideoWriter object
    video_writer.release()
    
    print(f"Video saved as {video_filename}")