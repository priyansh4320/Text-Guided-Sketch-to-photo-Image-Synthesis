import torch 
from torchvision import transforms 
import os
import numpy as np
from PIL import Image

#start class---------------------------------------------------------------

class process_images_and_sketches:
    def read_images_files(self,folder_path):
        folder = os.listdir(folder_path)
        images = []
        for files in folder:
            image = Image.open(os.path.join(folder_path, files))
            # Convert the image to a tensor.
            tensor = transforms.ToTensor()(image)
            images.append(image) 
        return np.array(images)

#end class---------------------------------------------------------------------



#driver code---------------------------------------------------------------
if __name__=="__main__":
    folder_path = 'clone\Text-Guided-Sketch-to-photo-Image-Synthesis\images'
    obj = process_images_and_sketches()
    image_tensor = obj.read_images_files(folder_path)
    print(image_tensor[:3])
    print(type(image_tensor[0]))
    pass