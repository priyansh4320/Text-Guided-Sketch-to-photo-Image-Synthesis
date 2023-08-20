import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer

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
    

    def tokenize_and_extract_word_embeddings(self,text_arr):
        #load model
        model = SentenceTransformer('stsb-roberta-base')
        result_embeddings = []
        for text in text_arr:
            embeddings = model.encode(text, convert_to_tensor=True)
            result_embeddings.append(embeddings)
        return np.array(result_embeddings)
#end of class-----------------------------------------------------------------------------------------


if __name__ == '__main__':
    folder_path = 'clone\Text-Guided-Sketch-to-photo-Image-Synthesis\Text'
    obj = preprocess_data()
    text_arr = obj.read_text_file(folder_path)
    embeddings = obj.tokenize_and_extract_word_embeddings(text_arr[0:3])


    print(text_arr[0:3],'\n',embeddings)
    print("p")
    pass