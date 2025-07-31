from sklearn.metrics import confusion_matrix
from dataloaders import RealDatasets
from torch.utils.data import DataLoader
from classifiers import Classifier1
import torch
import torchvision
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

if __name__== "__main__":
    test_csv_path = "test_gender_sex.csv"
    GPU_TO_USE="0"
    device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"
    test_dataset = RealDatasets(test_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers= 8)
    PATH = "/nas-ctm01/homes/mlsampaio/classifier/model_weights_classifier_29_05.pth"
    model = Classifier1(128).to(device)
    model.load_state_dict(torch.load(PATH))

    
    total_images = 1001349
    weights_sex = torch.tensor([10, total_images/680923, total_images/317775]).to(device)
    loss_sex = nn.CrossEntropyLoss(weight=weights_sex) 
    loss_ethnicity = nn.CrossEntropyLoss() 

    model.eval()
    test_total_loss = []
     
     #test_total = 0
   #   test_acc_sex_epoch = 0
   #   test_acc_ethnicity_epoch = 0

    true_labels_sex = [] 
    predicted_labels_sex = []
        
     #saving the labels for the metrics per batch (F1-score, balanced and normal accuracy)
    true_labels_ethnicity = [] 
    predicted_labels_ethnicity = []
        
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):

            X = X.float().to(device)

            ypred = model(X)

            y_sex = y["sex"].to(device)
            y_ethnicity = y["ethnicity"].to(device)

            test_loss = 0
            
            sex_loss = loss_sex(ypred[0], y_sex)  #compute the loss for the sex 
            ethnicity_loss = loss_ethnicity(ypred[1], y_ethnicity) #compute the loss for the ethnicity
            
            alpha = 0.5 #This value defines the contribution of each loss 
            test_loss = alpha*sex_loss + alpha*ethnicity_loss

            test_total_loss.append(test_loss.item())
            
            # Accumulate true and predicted labels
            true_labels_sex.extend(y_sex.cpu().numpy())
            print(len(true_labels_sex))
            predicted_labels_sex.extend(torch.argmax(ypred[0], axis=1).detach().cpu().numpy())
            print(len(predicted_labels_sex))
            
            # Accumulate true and predicted labels
            true_labels_ethnicity.extend(y_ethnicity.cpu().numpy())
            print(len(true_labels_ethnicity))
            predicted_labels_ethnicity.extend(torch.argmax(ypred[1], axis=1).detach().cpu().numpy())
            print(len(predicted_labels_ethnicity))
    
    cf_matrix1 = confusion_matrix(true_labels_sex, predicted_labels_sex)
    print(cf_matrix1)
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm1))
    disp.plot(cmap=plt.cm.Blues)  # You can choose different color maps
    plt.savefig("cm1.png")
    cf_matrix2 = confusion_matrix(true_labels_ethnicity, predicted_labels_ethnicity)
    print(cf_matrix2)
    plt.savefig("cm2.png")
    

