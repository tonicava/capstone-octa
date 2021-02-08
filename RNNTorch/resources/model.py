import torch.nn.functional as F
import torch.nn as nn


class Regression(nn.Module):
    """
    We define a neural network that performs linear regression.
    The network will accept the given features as input, and produce 
    a float value that is the price prediction.
    
    We use RMSELoss to train the network
    """

    def __init__(self, input_features, hidden_dim1, hidden_dim2, output_dim):
        """
        We initialize the model by setting up linear layers.
        :param input_features: the number of input features in the training/test data
        :param hidden_dim1: helps define the number of nodes in the 1st hidden layer
        :param output_dim: the number of outputs we want to produce
        """
        super(Regression, self).__init__()

        self.fc1 = nn.Linear(input_features, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.drop = nn.Dropout(0.3)

    
    def forward(self, x):
        """
        Performs a forward pass of the model on the input features x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A float value that is the price prediction
        """
        out = F.relu(self.fc1(x))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.drop(out)
        out = self.fc3(out)
        return out
        
