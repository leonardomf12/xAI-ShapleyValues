from dataloaders import RealDatasets, split_data
import pandas as pd 
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt 

""" Code to extract the data distribution in the training and testing datasets """

def create_list(loader):
    sex =[]
    ethnicity=[]
    for i, (X, y) in enumerate(loader): # loop para treino

        start_time = time.time()
        sex.extend(y["sex"])
        #print(y["sex"])
        #print(y["ethnicity"])
        ethnicity.extend(y["ethnicity"].cpu().numpy())
        end_time = time.time()
        batch_time = end_time - start_time
        print(f"Batch {i+1}: Time taken: {batch_time:.4f} seconds")

    return sex, ethnicity

if __name__== "__main__":
    # sex = y["sex"]
    # ethnicity = y["ethnicity"]
    
    dataset_folder = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_emb/"
    train_paths, val_paths, test_paths = split_data(dataset_folder, 0.1, 0.1)
    

    dataset_path = "/nas-ctm01/homes/mlsampaio/classifier/maad_full_attributes.csv"
    train_dataset = RealDatasets(train_paths, dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers = 8)
    
    #dataloader for validation 
    val_dataset = RealDatasets(val_paths, dataset_path)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers = 8)
   
    # #dataloader for testing
    test_dataset = RealDatasets(test_paths, dataset_path)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    
    
    start_time = time.time()
    train_sex_list, train_et_list = create_list(train_loader)
    end_time = time.time()
    time_process = end_time - start_time
    print(f'\t time for processing {time_process}')

    print(train_sex_list, len(train_sex_list))
    print(train_et_list)
    sex=[0,1,2]
    plt.pie(train_sex_list)
    plt.savefig("train_sex_pie.png")
    
    ethnicity=[0,1,2,3]
    plt.pie(train_et_list)
    plt.savefig("train_et_pie.png")
    # train_dic = create_dic(train_loader)
    # df1 = pd.DataFrame(data=train_dic)
    
    val_sex_list, val_et_list = create_list(val_loader)
    plt.pie(val_sex_list)
    plt.savefig("val_sex_pie.png")
    plt.pie(val_et_list)
    plt.savefig("val_et_pie.png")
    # val_dic = create_dic(val_loader)
    # df2 = pd.DataFrame(data=val_dic)
    
    test_sex_list, test_et_list = create_list(test_loader)
    plt.pie(test_sex_list)
    plt.savefig("val_sex_pie.png")
    plt.pie(test_et_list)
    plt.savefig("val_et_pie.png")
    # print("yey")
    # test_dic = create_dic(test_loader)
    # df3 = pd.DataFrame(data=test_dic)

    # print(df1.value_counts())
    # print(df2.value_counts())
    # print(df3.value_counts())
    # print(y_e)
    # eyes = ["0", "1", "2"]
    # # Creating plot

    # plt.pie(y_eye, labels = eyes)
    
    
    # labels_sex=[0,1,2]
    # labels_ethnicity=[0,1,2,3]
    # # Create a Pandas Series from the labels
    # labels_sex = pd.Series(labels_sex)

    # Use value_counts() to count the occurrences of each label
    # label_counts = train_dataset.value_counts()

    # print(label_counts )

