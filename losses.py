import IPython
import torch
import torch.nn as nn


class MTL2Loss(nn.Module):
    def __init__(self, scale_factor=1.0):
        super(MTL2Loss, self).__init__()

    def forward(self, predictions, targets):
        print("inside loss")
        IPython.embed()

        # Custom loss logic
        loss = torch.mean((predictions - targets) ** 2)  # For example, MSE loss

        # If there are learnable parameters, use them in the computation
        loss *= self.scale_factor

        return loss

    def reset_parameters(self):
        """
        Reset the parameters of the loss function (if applicable).
        """
        # Reset the scale factor or any other parameters
        self.scale_factor.data.fill_(1.0)

