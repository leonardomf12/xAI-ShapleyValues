from train_eval import train_val, test_model
from classifiers import Classifier1, SexClassifier, EthnicityClassifier
import sys
import numpy as np
from classifiers import Classifier1, SexClassifier, EthnicityClassifier
from dataloaders import RealDatasets, split_data, join_csv, write_files_path, read_files_path, SexDataset, EthnicityDataset
from torch.utils.data import DataLoader
import wandb
import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import time 

#Folder path for all the dataset 
dataset_folder = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_emb/"


#    #spliting the data by identity - returning a list with all the files for each dataset
train_paths, val_paths, test_paths = split_data(dataset_folder, 0.8, 0.1)

#save this lists in txt files bc otherwise it would take a long time doing this everytime we run the code.
#files_path(train_paths, val_paths, test_paths)
   
print("no. images train:", len(train_paths))
print("no. images val:", len(val_paths))
print("no. images test:", len(test_paths))

#print(train_paths)
#    #create dataloader for training
dataset_path = "/nas-ctm01/homes/mlsampaio/classifier/maad_gender.csv"
train_dataset = SexDataset(train_paths, dataset_path)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers = 8)
  
 
#dataloader for validation 
val_dataset = SexDataset(val_paths, dataset_path)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers = 8)
   
#dataloader for testing
test_dataset = SexDataset(test_paths, dataset_path)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers= 8)

print("no. batches train", len(train_loader))
print("no. batches val", len(val_loader))
print("no. batches test", len(test_loader))


# Train the network and test 
INPUT_DIM = 128 #for the model of FaceNet the dimension is 128
N_EPOCHS = 5 #vou começar com poucas épocas e depois aumentar para 50, 75 e 100

#Defining to use GPU unless it is not available
GPU_TO_USE="0"
DEVICE = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
   
#Defining the classifier 
model = Classifier1(INPUT_DIM)
model1 = SexClassifier(INPUT_DIM)
model2 = EthnicityClassifier(INPUT_DIM)

# Define the loss function and optimizer
loss = nn.CrossEntropyLoss() #Lets start with a 
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3) 

model1 = model1.to(DEVICE)
   
#    # start a new wandb run to track this script
wandb.init(
   # set the wandb project where this run will be logged
   project="Classifier_Ethinicity",

   # track hyperparameters and run metadata
   config={
    "learning_rate": 1e-3,
    "architecture": "ClassifierSex",
    "embedding": "FaceNet",
    "embedding_size": 128,
    "loss": "Cross Entropy",
    "Optimizer": "Adam",
    "epochs": 5,
    "batch_size": 128, 
    }
   )

train_loss, train_acc_sex, train_acc_ethnicity, train_acc_sex_balanced, train_acc_ethnicity_balanced, valid_hist, valid_acc_sex, valid_acc_ethnicity, valid_acc_sex_balanced, valid_acc_ethnicity_balanced= train_val(model, loss, optimizer, train_loader, val_loader, N_EPOCHS, DEVICE)
   
# Save model weights
torch.save(model1.state_dict(), 'model_weights_classifier1.pth')
test_loss, acc_sex_epoch, balanced_acc_sex_epoch, acc_ethnicity_epoch, balanced_acc_ethnicity_epoch = test_model(model1, loss, test_loader, DEVICE)
print('Test Loss.:' ,test_loss, "Test Sex Acc.:", acc_sex_epoch, "Balanced Test Sex Acc.:", balanced_acc_sex_epoch, "Test Ethinicty Acc.:", acc_ethnicity_epoch,"Balanced Test Ethnicity Acc.:", balanced_acc_ethnicity_epoch)
