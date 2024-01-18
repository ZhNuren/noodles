import os
from PIL import Image
from tqdm import tqdm
def resize_images(input_folder, output_folder, size=(500, 500)):
    for root, dirs, files in tqdm(os.walk(input_folder)):
        for file in files:
            file_path = os.path.join(root, file)
 
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Read and resize the image using bilinear interpolation
                with Image.open(file_path) as img:
                    resized_img = img.resize(size, Image.BILINEAR)
 
                # Create the corresponding subfolder structure in the output folder
                relative_path = os.path.relpath(file_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
 
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
                # Save the resized image in the output folder
                resized_img.save(output_path)
 
if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "/home/nuren.zhaksylyk/Documents/Research/noodles/dataset/isic/train/"
    output_folder = "/home/nuren.zhaksylyk/Documents/Research/noodles/dataset/resized/"
 
    # Specify the desired size for the resized images
    target_size = (224, 224)
 
    # Call the function to resize images and save in the output folder
    resize_images(input_folder, output_folder, size=target_size)