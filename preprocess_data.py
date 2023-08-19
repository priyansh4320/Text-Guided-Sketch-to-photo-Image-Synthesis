import torch
import numpy as np
import pandas as pd
import os

#start class----------------------------------------------------------------------------------------------
class preprocess_data:
    def __init__(self) -> None:   
        pass
    #read all txt file and store in numpy array
    def read_text_file(self,folder_path):

        txt_folder = os.listdir(folder_path)
        texts = []
        
        for  text_file in txt_folder:
            texts.append(np.loadtxt(os.path.join(folder_path,text_file)))   
        
        return np.concatenate(texts)
#end of class-----------------------------------------------------------------------------------------


if __name__ == '__main__':
    folder_path = 'clone\Text-Guided-Sketch-to-photo-Image-Synthesis\Text'
    obj = preprocess_data()
    text_arr = obj.read_text_file(folder_path)
    print(text_arr)
    pass