import sys
import numpy as np
from classifiers import Classifier1
from dataloaders import RealDatasets, split_data
from torch.utils.data import DataLoader
import wandb
import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import time
from sklearn.utils.class_weight import compute_class_weight
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import classification_report


# from torch.optim.lr_scheduler import StepLR

def train_val(model, loss_sex, loss_ethnicity, optimizer, train_loader, val_loader, n_epochs, device, patience=np.inf):
    # repeat training for the desired number of epochs
    train_hist = []

    valid_hist = []
    valid_acc_sex = []
    valid_acc_ethnicity = []
    valid_acc_sex_balanced = []
    valid_acc_ethnicity_balanced = []
    train_acc_sex = []
    train_acc_ethnicity = []
    train_acc_sex_balanced = []
    train_acc_ethnicity_balanced = []
    valid_acc_sex_balanced = []
    valid_acc_ethnicity_balanced = []

    best_valid_loss = None
    plateau = 0

    print(model)  # Print the structure of the model

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        #   train_total = 0

        # initialize before the iterations for each epoch in each epoch it will interate in the batches
        total_train_loss = []
        true_labels_sex = []
        predicted_labels_sex = []

        # saving the labels for the metrics per batch (F1-score, balanced and normal accuracy)
        true_labels_ethnicity = []
        predicted_labels_ethnicity = []

        # training loop
        model.train()  # set model to training mode (affects dropout and batch norm.)
        start_time_epoch = time.time()
        for i, (X, y) in enumerate(train_loader):  # loop para treino
            start_time = time.time()
            X = X.float().to(device)  # Convert the embeddings to float and then put in the device

            ypred = model(X)  # forward pass #prevision from the model - has two heads so two dimensions
            # print("ypred:", ypred)
            y_sex = y["sex"].to(device)  # label for sex
            # print("y_sex", y_sex)
            y_ethnicity = y["ethnicity"].to(device)  # label for ethnicity
            # print("y_ethnicity", y_ethnicity)

            sex_loss = loss_sex(ypred[0], y_sex)  # compute the loss for the sex
            # print("train sex loss", sex_loss)
            # print(sex_loss)
            # print("sex_loss", sex_loss)
            ethnicity_loss = loss_ethnicity(ypred[1], y_ethnicity)  # compute the loss for the ethnicity
            # print("train ethinicity loss", ethnicity_loss)
            # print("ethnicity loss", ethnicity_loss)

            alpha = 0.5  # This value defines the contribution of each loss
            train_loss = alpha * sex_loss + alpha * ethnicity_loss

            optimizer.zero_grad()  # set all gradients to zero (otherwise they are accumulated)
            train_loss.backward()  # backward pass (i.e. compute gradients)
            optimizer.step()  # update the parameters

            total_train_loss.append(train_loss.detach().cpu().numpy())

            # #Number of correct predictions per batch 
            # train_corrects_sex, train_corrects_ethnicity = 0,0 
            # train_corrects_sex += (torch.argmax(ypred[0], 1) == y_sex).float().sum() 
            # train_corrects_ethnicity += (torch.argmax(ypred[1], 1) == y_ethnicity).float().sum()

            # Accumulate true and predicted labels
            true_labels_sex.extend(y_sex.cpu().numpy())
            # print(".......")
            # print(y_sex)
            predicted_labels_sex.extend(torch.argmax(ypred[0], axis=1).detach().cpu().numpy())
            # print(ypred[0])
            # print(torch.argmax(ypred[0], axis=1))

            # Accumulate true and predicted labels
            true_labels_ethnicity.extend(y_ethnicity.cpu().numpy())
            # print(y_ethnicity)
            predicted_labels_ethnicity.extend(torch.argmax(ypred[1], axis=1).detach().cpu().numpy())
            # print(ypred[1])
            # print(torch.argmax(ypred[1], axis=1))

            end_time = time.time()
            batch_time = end_time - start_time
            # scheduler.step()
            # print(f'\t Batch {i+1} Train Loss: {train_loss.item():.3f} | Sex Acc: {acc_sex*100:.2f}%  | Ethnicity Acc: {acc_ethnicity*100:.2f}%')
            # print(f"\t Batch {i+1}: Time taken: {batch_time:.4f} seconds")

        acc_sex_epoch = accuracy_score(true_labels_sex, predicted_labels_sex)
        balanced_acc_sex_epoch = balanced_accuracy_score(true_labels_sex, predicted_labels_sex)

        acc_ethnicity_epoch = accuracy_score(true_labels_ethnicity, predicted_labels_ethnicity)
        balanced_acc_ethnicity_epoch = balanced_accuracy_score(true_labels_ethnicity, predicted_labels_ethnicity)

        train_report_sex = classification_report(true_labels_sex, predicted_labels_sex, labels=[0, 1])
        print("train report sex", train_report_sex)
        train_report_ethnicity = classification_report(true_labels_ethnicity, predicted_labels_ethnicity,
                                                       labels=[0, 1, 2, 3])
        print("train report ethnicity", train_report_ethnicity)

        train_acc_sex.append(acc_sex_epoch)
        train_acc_ethnicity.append(acc_ethnicity_epoch)
        train_hist.append(np.mean(total_train_loss))
        train_acc_sex_balanced.append(balanced_acc_sex_epoch)
        train_acc_ethnicity_balanced.append(balanced_acc_ethnicity_epoch)

        val_loss, val_acc_sex, val_balanced_acc_sex, val_acc_ethnicity, val_balanced_acc_ethnicity = test_model(model,
                                                                                                                loss_sex,
                                                                                                                loss_ethnicity,
                                                                                                                val_loader,
                                                                                                                DEVICE)
        valid_hist.append(val_loss)
        valid_acc_sex.append(val_acc_sex)
        valid_acc_ethnicity.append(val_acc_ethnicity)
        valid_acc_sex_balanced.append(val_balanced_acc_sex)
        valid_acc_ethnicity_balanced.append(val_balanced_acc_ethnicity)

        # scheduler.step(val_loss)
        # end_time_epoch = time.time()
        # print("Epoch:", i)
        # print(f'\t Time Epoch: {end_time_epoch-start_time_epoch}')
        # print(f'\tEPOCH Train Loss: {(np.mean(total_train_loss)).item():.3f} | Sex Acc: {acc_sex_epoch:.2f}  | Ethnicity Acc: {acc_ethnicity_epoch:.2f}| Sex Balanced Acc  {balanced_acc_sex_epoch:.2f}| Ethnicity Balanced Acc {balanced_acc_ethnicity_epoch:.2f}')
        # print(f'\tEPOCH Validation Loss: {(val_loss).item():.3f} | Sex Acc: {val_acc_sex:.2f}  | Ethnicity Acc: {val_acc_ethnicity:.2f}| Sex Balanced Acc  {val_balanced_acc_sex:.2f}%| Ethnicity Balanced Acc  {val_balanced_acc_ethnicity:.2f}')
        # #stream metrics during training
        wandb.log({"acc_sex_train": acc_sex_epoch, "balanced_acc_sex_train": balanced_acc_sex_epoch,
                   "acc_ethnicity_train": acc_ethnicity_epoch,
                   "balanced_acc_ethnicity_train": balanced_acc_ethnicity_epoch,
                   "train loss": np.mean(total_train_loss), "validation loss": val_loss,
                   "validation acc sex": val_acc_sex, "validation acc ethnictiy": val_acc_ethnicity,
                   "validation balanced acc sex": val_balanced_acc_sex,
                   "validation balanced acc ethnicity": val_balanced_acc_ethnicity})

        if best_valid_loss is None:
            best_valid_loss = val_loss
        # Saving the best model so far:
        elif val_loss < best_valid_loss:
            best_valid_loss = val_loss
            plateau = 0
        else:
            plateau += 1
            if plateau >= patience:
                print('.... Early stopping the train.')
                return train_loss, train_acc_sex, train_acc_ethnicity, train_acc_sex_balanced, train_acc_ethnicity_balanced, valid_hist, valid_acc_sex, valid_acc_ethnicity, valid_acc_sex_balanced, valid_acc_ethnicity_balanced

    return train_loss, train_acc_sex, train_acc_ethnicity, train_acc_sex_balanced, train_acc_ethnicity_balanced, valid_hist, valid_acc_sex, valid_acc_ethnicity, valid_acc_sex_balanced, valid_acc_ethnicity_balanced


def test_model(model, loss_sex, loss_ethnicity, test_loader, device):
    model.eval()
    test_total_loss = []

    # test_total = 0
    #   test_acc_sex_epoch = 0
    #   test_acc_ethnicity_epoch = 0

    true_labels_sex = []
    predicted_labels_sex = []

    # saving the labels for the metrics per batch (F1-score, balanced and normal accuracy)
    true_labels_ethnicity = []
    predicted_labels_ethnicity = []

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X = X.float().to(device)

            ypred = model(X)

            y_sex = y["sex"].to(device)
            y_ethnicity = y["ethnicity"].to(device)

            test_loss = 0

            sex_loss = loss_sex(ypred[0], y_sex)  # compute the loss for the sex
            # print("val sex loss", sex_loss)
            ethnicity_loss = loss_ethnicity(ypred[1], y_ethnicity)  # compute the loss for the ethnicity
            # print("val ethinicity loss", ethnicity_loss)

            alpha = 0.5  # This value defines the contribution of each loss
            test_loss = alpha * sex_loss + alpha * ethnicity_loss

            test_total_loss.append(test_loss.item())

            # Accumulate true and predicted labels
            true_labels_sex.extend(y_sex.cpu().numpy())
            predicted_labels_sex.extend(torch.argmax(ypred[0], axis=1).detach().cpu().numpy())

            # Accumulate true and predicted labels
            true_labels_ethnicity.extend(y_ethnicity.cpu().numpy())
            predicted_labels_ethnicity.extend(torch.argmax(ypred[1], axis=1).detach().cpu().numpy())

        acc_sex_epoch = accuracy_score(true_labels_sex, predicted_labels_sex)
        balanced_acc_sex_epoch = balanced_accuracy_score(true_labels_sex, predicted_labels_sex)

        acc_ethnicity_epoch = accuracy_score(true_labels_ethnicity, predicted_labels_ethnicity)
        balanced_acc_ethnicity_epoch = balanced_accuracy_score(true_labels_ethnicity, predicted_labels_ethnicity)

        val_report_sex = classification_report(true_labels_sex, predicted_labels_sex, labels=[0, 1])
        print("val or test report train")
        print(val_report_sex)
        val_report_ethnicity = classification_report(true_labels_ethnicity, predicted_labels_ethnicity,
                                                     labels=[0, 1, 2, 3])
        print("val or test report train")
        print(val_report_ethnicity)

    return np.mean(
        test_total_loss), acc_sex_epoch, balanced_acc_sex_epoch, acc_ethnicity_epoch, balanced_acc_ethnicity_epoch


if __name__ == "__main__":
    # Folder path for all the dataset
    dataset_folder = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_emb/"

    #    african_csv = "/nas-ctm01/homes/mlsampaio/classifier/maad_dataset_attributes_balancedFace_African.csv"
    #    asian_csv = "/nas-ctm01/homes/mlsampaio/classifier/maad_dataset_attributes_balancedFace_Asian.csv"
    #    caucasian_csv = "/nas-ctm01/homes/mlsampaio/classifier/maad_dataset_attributes_balancedFace_Caucasian.csv"
    #    indian_csv = "/nas-ctm01/homes/mlsampaio/classifier/maad_dataset_attributes_balancedFace_Indian.csv"

    #    csv_file = join_csv(african_csv, asian_csv, caucasian_csv, indian_csv) #joining all the information in one dataframe

    #    #spliting the data by identity - returning a list with all the files for each dataset
    #  train_paths, val_paths, test_paths = split_data(dataset_folder, 0.8, 0.1)

    #  print("no. images train:", len(train_paths))
    #  print("no. images val:", len(val_paths))
    #  print("no. images test:", len(test_paths))

    # print(train_paths)
    #    #create dataloader for training
    dataset_path = "/nas-ctm01/homes/mlsampaio/classifier/maad_gender.csv"

    train_csv_path = "train_gender_sex.csv"
    val_csv_path = "val_gender_sex.csv"
    test_csv_path = "test_gender_sex.csv"

    train_dataset = RealDatasets(train_csv_path)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

    # dataloader for validation
    val_dataset = RealDatasets(val_csv_path)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

    # dataloader for testing
    test_dataset = RealDatasets(test_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

    print("no. batches train", len(train_loader))
    print("no. batches val", len(val_loader))
    print("no. batches test", len(test_loader))

    # Train the network and test s
    INPUT_DIM = 128  # for the model of FaceNet the dimension is 128
    N_EPOCHS = 100  # vou começar com poucas épocas e depois aumentar para 50, 75 e 100
    LEARNING_RATE = 1e-4

    # Defining to use GPU unless it is not available
    GPU_TO_USE = "0"
    DEVICE = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"

    # Defining the classifier
    model = Classifier1(INPUT_DIM)
    # model1 = SexClassifier(INPUT_DIM)
    # model2 = EthnicityClassifier(INPUT_DIM)
    #  total_images = 1
    #  factor = 0.9
    # weights_sex = torch.tensor().to(DEVICE)
    # weights_sex = torch.tensor([(total_images/(2651))*(0.9*0.9*0.9),total_images/(680923), total_images/(317775)]).to(DEVICE)
    # Define the loss function and optimizer
    loss_sex = nn.CrossEntropyLoss()
    loss_ethnicity = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    model = model.to(DEVICE)

    #    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Classifier without mode",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
            "architecture": "64 (64 output)",
            "embedding": "FaceNet",
            "embedding_size": 128,
            "loss": "Cross Entropy",
            "Optimizer": "Adam",
            "epochs": 100,
            "batch_size": 128,
            "dropout": 0,
            "learning rate decay": "False",
            "batch normalization": "True, Standard Values",
        }
    )

train_loss, train_acc_sex, train_acc_ethnicity, train_acc_sex_balanced, train_acc_ethnicity_balanced, valid_hist, valid_acc_sex, valid_acc_ethnicity, valid_acc_sex_balanced, valid_acc_ethnicity_balanced = train_val(
    model, loss_sex, loss_ethnicity, optimizer, train_loader, val_loader, N_EPOCHS, DEVICE)

# Save model weights
torch.save(model.state_dict(), 'model_weights_classifier_01_06_baseline.pth')
test_loss, acc_sex_epoch, balanced_acc_sex_epoch, acc_ethnicity_epoch, balanced_acc_ethnicity_epoch = test_model(model,
                                                                                                                 loss_sex,
                                                                                                                 loss_ethnicity,
                                                                                                                 test_loader,
                                                                                                                 DEVICE)
print('Test Loss.:', test_loss, "Test Sex Acc.:", acc_sex_epoch, "Balanced Test Sex Acc.:", balanced_acc_sex_epoch,
      "Test Ethinicty Acc.:", acc_ethnicity_epoch, "Balanced Test Ethnicity Acc.:", balanced_acc_ethnicity_epoch)
