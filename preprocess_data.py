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
            file = open(os.path.join(folder_path,text_file),'r')
            texts.append(" ".join(file.readlines()))   
        
        return np.asarray(texts)
    
    def remove_punc(self,text_arr):
        punc = ",.?/;:'[{]()}"""
        res_arr = []
        for text in text_arr:
            text = "".join([i for i in text if i not in punc])
            res_arr.append(" ".join(text.split('\n')))
        return np.asarray(res_arr)

    def tokenize_and_extract_word_embeddings(self,text_arr):
        return
#end of class-----------------------------------------------------------------------------------------


if __name__ == '__main__':
    folder_path = 'clone\Text-Guided-Sketch-to-photo-Image-Synthesis\Text'
    obj = preprocess_data()
    text_arr = obj.read_text_file(folder_path)
    text = obj.remove_punc(text_arr[0:3])


    print(text_arr[0:3],'\n',text)

    pass