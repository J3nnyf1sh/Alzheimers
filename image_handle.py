import os 
from PIL import Image, ImageOps

"""
    Resizes the uploaded image to a square
"""

def pad(imagePath):
    image = Image.open(imagePath)
    width, height = image.size
    padding = (0,0,0,0) #top, right, bottom, left

    if width > height:
        padding = (0, (width - height) // 2, 0, (width - height)//2)
    elif height > width:
        padding = (0, (height - width) // 2, 0, (height - width)//2)
    padded = ImageOps.expand(image, padding, 0)
    return padded

def resize(image, size = (512, 512)):
    resizedImage = image.resize(size)
    return resizedImage

def process_image(inputFolder, outputFolder):
    for image in os.listdir(inputFolder):
        if image.endswith(('.jpg', '.png')):
            imagePath = os.path.join(inputFolder, image)
            
            img = Image.open(imagePath)

            width, height = img.size

            if width != height:
                img = pad(imagePath)
            
            resizedImage = resize(img) 
            resizedImage.save(os.path.join(outputFolder, image))
            print("done")

    print("done")
            

process_image('datasets/old/train/images', 'datasets/brain_tumour/train/images')
process_image('datasets/old/valid/images', 'datasets/brain_tumour/valid/images')