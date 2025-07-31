import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLossWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.loss = None
        self.base_model = base_model

        self.head_names = ["sex", "race", "square", "eyeglasses", "nose"]
        self.head_weights = nn.Parameter(
            torch.ones(len(self.head_names)) + torch.normal(mean=0.0, std=0.01, size=(len(self.head_names),))
        )
        # self.loss_sex = nn.CrossEntropyLoss()
        # self.loss_race = nn.CrossEntropyLoss()
        # self.loss_square_face = nn.BCEWithLogitsLoss()
        # self.loss_eyeglasses = nn.BCEWithLogitsLoss()
        # self.loss_pointy_nose = nn.BCEWithLogitsLoss()
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCEWithLogitsLoss()


    def forward(self, x_train, targets=None):

        x_out = self.base_model(x_train)
        y_true = self.prepoc_targets(targets)

        self.loss = 0.0
        for w_i, head_name in enumerate(self.head_names):

            x = x_out[head_name]
            y = y_true[head_name]

            if head_name in self.head_names[-3:]: #BCE
                y = y.view(-1, 1)

                # Masking unknowns
                unk_mask = ~(y == -1)
                x = x[unk_mask]
                y = y[unk_mask]

                loss_head = self.BCELoss(x, y)
            else:

                unk_mask = ~(y == -1).any(dim=1)
                x = x[unk_mask]
                y = y[unk_mask]
                y = y.argmax(dim=1) # Converting from one-hot vectors to class indices

                loss_head = self.CELoss(x, y)

            self.loss += self.head_weights[w_i] * loss_head


        return self.loss
        #mask = (targets != -1)
        # TODO do a for loop per head -> remove all the -1 within the batch -> calculate loss -> reconstruct initial shape -> perform mean or sum to batch.

    @staticmethod
    def prepoc_targets(targets):
        y_true = {}

        y_true["sex"] = targets[:, :1 + 1]
        y_true["race"] = targets[:, 2:5 + 1]
        y_true["square"] = targets[:, -3]
        y_true["eyeglasses"] = targets[:, -2]
        y_true["nose"] = targets[:, -1]

        return y_true
