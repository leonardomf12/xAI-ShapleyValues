import os
import pandas as pd
import numpy as np
import torch
import torchvision
import random 
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

""" 
Class to generate the dataloaders for train, val, test
Inputs: CSV with columns for the path of the embedding path,sex,ethnicity
"""
class RealDatasets(Dataset): # definning to read our image and annotations
    def __init__(self, annotations):
        self.annotations = pd.read_csv(annotations)
        self.annotations = self.annotations[self.annotations["sex"]!=0] #remove the lines with label zero
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
           embedding_path = self.annotations.iloc[idx,0] #goes through the embedding list
           #print(embedding_path)
           embedding = torch.from_numpy(np.load(embedding_path)) #loads the embbeding
           converting = {1:1, -1:0} 
           race = {"African":0, "Asian":1, "Caucasian":2, "Indian":3} 
           
           annotation_sex = self.annotations.iloc[idx,1]
           
           sex=converting[annotation_sex]
           ethnicity = race[embedding_path.split('/')[-3]] 

           annotations = {'sex': sex,  'ethnicity': ethnicity}
           return embedding, annotations
    
""" Function created to split the data into train, validation and test according to identity
@Parameters:
dir - directory of the dataset folder 
train_ratio - percentage of the dataset that is going to be used for train
val_ratio - "" for validation
test_ratio - "" for testing
"""
def split_data(dir, train_ratio, val_ratio):
   ids = []
   for path, subdirs, files in os.walk(dir):
        if not subdirs:
            ids.append(path) #retorna todos as folders ou seja os diferentes ids.
    
   n_folders = len(ids)
   train_idx = int(train_ratio*n_folders)
   print("train ids:", train_idx)
   val_idx = train_idx + int(val_ratio*n_folders)
   print("val ids:", val_idx)
   
   #shuffle the folders for the etnicities to be more homogeneous
   random.shuffle(ids)

   train_folders= ids[0:train_idx]
   val_folders = ids[train_idx:val_idx]
   test_folders = ids[val_idx:]
   print("test ids:", len(test_folders))
   train_paths = []
   for folder in train_folders:
       folder_path = os.path.join(dir, folder) #junta as pastas
       files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)] #junta todos os ficheiros para aquela identidade
       train_paths.extend(files) #adiciona os files à lista de treino.
    
   val_paths = []
   for folder in val_folders:
       folder_path = os.path.join(dir, folder) #junta as pastas
       files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)] #junta todos os ficheiros para aquela identidade
       val_paths.extend(files) #adiciona os files à lista de treino.
   
   test_paths = []
   for folder in test_folders:
       folder_path = os.path.join(dir, folder) #junta as pastas
       files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)] #junta todos os ficheiros para aquela identidade
       test_paths.extend(files) #adiciona os files à lista de treino.
    
   return train_paths, val_paths, test_paths


#Function to join the csv from the different ethnicitys
#Returns a single csv with all the csv joined 
def join_csv(african_csv, asian_csv, caucasian_csv, indian_csv):
       african_csv = pd.read_csv(african_csv)
       asian_csv = pd.read_csv(asian_csv)
       caucasian_csv = pd.read_csv(caucasian_csv)
       indian_csv = pd.read_csv(indian_csv)
       full_csv = pd.concat([african_csv, asian_csv, caucasian_csv, indian_csv])
       print(full_csv)
       full_csv.to_csv("maad_full_attributes.csv", index=True)

       return full_csv

#Function to create the CSV with the features that are needed   
def create_csv(path_list, csv_path, new_csv_path):
    csv_file = pd.read_csv(csv_path)
    csv_file = csv_file.drop(columns=["asian","white","black","attractive"])
    with open(new_csv_path, mode='w', newline='') as file:
     writer = csv.writer(file)
     print("First")
     writer.writerow(["embedding_path", "label_ethnicity", "sex","young","middle_aged","senior","rosy_cheeks","shiny_skin","bald","wavy_hair","receding_hairline","bangs","sideburns","black_hair","blond_hair","brown_hair","gray_hair","no_beard","mustache,o_clock_shadow","goatee","oval_face","square_face","round_face","double_chain","high_cheekbones","chubby","obstructed_forehead","fully_visible_forehead","brown_eyes","bags_under_eyes","bushy_eyebrows","arched_eyebrows","mouth_closed","smiling","big_lips","big_nose","pointy_nose","heavy_makeup","wearing_hat","wearing_earrings","wearing_necktie","wearing_lipstick","no_eyewear","eyeglasses"])
     for embedding_path in path_list:
      print("2nd")
      image_path = str(str(embedding_path).replace(".npy", ".jpg")) #string from which we are going to search in the DataFrame 
      image_path = str(str(image_path).replace("race_per_7000_emb", "race_per_7000"))
      
      attributes = (csv_file[csv_file["path"].str.contains(image_path)])
      attributes = attributes.drop(columns="path")
      attributes = attributes.values
      attributes = str(attributes)
      attributes = list(attributes)
      attributes = [x for x in attributes if x in {"0", "1", "-1"}]
      #print(attributes)
      attributes.insert(0,embedding_path.split('/')[-3])
      attributes.insert(0, embedding_path)

      writer.writerow(attributes)
      print("ROW WRITTEN")

#run "dataloaders.py" to generate the CSV with the features that are needed
if __name__== "__main__":
    #dataset_folder = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_emb/"
    #csv_path = "/nas-ctm01/homes/mlsampaio/classifier/maad_full_attributes.csv"
    print("here")
    #train_paths, val_paths, test_paths = split_data(dataset_folder, 0.8, 0.1)
    #print("here1")

    train_csv = "/nas-ctm01/homes/mlsampaio/classifier/train_gender_sex.csv"
    val_csv = "/nas-ctm01/homes/mlsampaio/classifier/val_gender_sex.csv"
    test_csv = "/nas-ctm01/homes/mlsampaio/classifier/test_gender_sex.csv"

    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df["sex"]!=0]
    list_paths = train_df["path"]
    
    print("1")
    list_paths = [list_paths.values.tolist()[1].split("/")[7] for path in list_paths]
    print("2")
    for path in list_paths:
        print("3")
        filtered_df = train_df[train_df['path'].str.contains(path)]
        mode_value = filtered_df.mode(axis=0, numeric_only=True).iloc[0,0]
        train_df.loc[train_df['path'].str.contains(path), 'sex'] = mode_value

    print("3")
    train_df.to_csv("train_mode.csv")

    # print("here2")
    # # print("Debug1")
    # create_csv(train_paths, csv_path, train_csv)
    # print("here3")
    # create_csv(val_paths, csv_path, val_csv)
    # create_csv(test_paths, csv_path, test_csv)
    #train_csv= "/nas-ctm01/homes/mlsampaio/classifier/train_gender_sex.csv"
    # #Turn the csv into dataframes to get the statistics
    # print("Debug2")
    #train_df=pd.read_csv(train_csv)
    #converting = {0:0, 1:1, -1:2}
    # for idx in range(len(train_df)):
    #   embedding_path = train_df.iloc[idx,0] 
    #   print(embedding_path)
    
    #   annotation_sex = train_df.iloc[idx,1]
    # #   print("", annotation_sex)
    # #   annotation_sex=converting[annotation_sex]
    # #   print(annotation_sex)
    # val_df=pd.read_csv(val_csv)
    # test_df=pd.read_csv(test_csv)
    # print("Debug3")

    #  # print("Dataframes created")

    # train_sex_count = train_df["sex"].value_counts()
    # print(train_sex_count)
    # print(np.sum(train_sex_count))
    # # # #print(train_sex_count.values)
    # train_ethn_count = train_df["ethnicity"].value_counts()
    # print(train_ethn_count)
    # print(np.sum(train_ethn_count))
    # # # #print(train_ethn_count)
    # # # #saving figures for the count
    # labels_sex="Male","Female","Undefined"
    # labels_ethnicity="Asian", "African", "Caucasian", "Indian"
    
    # #Colors for a more aesthetic pie plot
    # colors = [ mcolors.CSS4_COLORS['royalblue'],  
    #       mcolors.CSS4_COLORS['cornflowerblue'],      
    #       mcolors.CSS4_COLORS['lightsteelblue'],
    #       mcolors.CSS4_COLORS['aliceblue'] ]    
    
    # # # values_sex_train = [680923,317775,2651]
    # # print("Starting train plots")
    # # plt.figure(figsize=(8, 5))
    # # plt.suptitle("Train Attributes Distribution", fontsize=16)
    # # plt.axis("off")
    # # plt.subplot(1,2,1)
    # # plt.pie(train_sex_count.values, labels=labels_sex, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
    # # plt.title("Sex", fontsize=15)
    # # plt.subplot(1,2,2)
    # # plt.pie(train_ethn_count.values, labels=labels_ethnicity, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
    # # plt.title("Ethinicity", fontsize=15)
    # # plt.tight_layout()

    # # plt.savefig("Trainplots.png")

    # val_sex_count = val_df["sex"].value_counts()
    # # # # print(val_sex_count)
    # # # values_sex_val = [84437,39061,283]
    # val_ethn_count = val_df["ethnicity"].value_counts()
    # # # #print(val_ethn_count)
    
    # # print("Starting val plots")
    # # plt.figure(figsize=(8, 5)) 
    # # plt.suptitle("Validation Attributes Distribution", fontsize=16)
    # # plt.axis("off")
    # # plt.subplot(1,2,1)
    # # plt.pie(val_sex_count.values, labels=labels_sex, colors=colors,autopct='%1.1f%%', textprops={'fontsize': 12})
    # # plt.title("Sex", fontsize=15)
    # # plt.subplot(1,2,2)
    # # plt.pie(val_ethn_count.values, labels=labels_ethnicity, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
    # # plt.title("Ethnicity", fontsize=15)
    # # plt.tight_layout()

    # # plt.savefig("Valplots.png")


    # test_sex_count = val_df["sex"].value_counts()
    # # # # print(test_sex_count)
    # test_ethn_count = val_df["ethnicity"].value_counts()
    # # # print(test_ethn_count)
    # print("Debug4")
    
    # print("Starting test plots")
    # plt.figure(figsize=(8, 5))
    # plt.suptitle("Test Attributes Distribution", fontsize=16)
    # plt.axis("off")
    # plt.subplot(1,2,1)
    # plt.pie(test_sex_count.values, labels=labels_sex, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
    # plt.title("Sex", fontsize=15)
    # plt.subplot(1,2,2)
    # plt.pie(test_ethn_count.values, labels=labels_ethnicity, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
    # plt.title("Ethinicity", fontsize=15)
    # plt.tight_layout()

    # plt.savefig("Testplots.png")
    
    # print("Train_size:", len(train_df))
    # print("Val size:", len(val_df))
    # print("Test size:", len(test_df))


    # print("Starting test plots")
    # plt.figure(figsize=(8, 5))
    # plt.suptitle("Attributes Distribution", fontsize=16)
    # plt.axis("off")
    # plt.subplot(1,2,1)
    # plt.pie(train_sex_count.values + test_sex_count.values + val_sex_count.values, labels=labels_sex, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
    # plt.title("Sex", fontsize=15)
    # plt.subplot(1,2,2)
    # plt.pie(train_ethn_count.values + val_ethn_count.values + test_ethn_count.values, labels=labels_ethnicity, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
    # plt.title("Ethinicity", fontsize=15)
    # plt.tight_layout()

    # plt.savefig("totalplots.png")
    

    
 

    
    


    


