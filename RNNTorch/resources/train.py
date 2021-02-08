import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from model import Regression


def model_fn(model_dir):
    """
    Loads the PyTorch model from the `model_dir` directory.
    
    :param model_dir: model directory
    :return: model created
    """
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Regression(model_info['input_features'], model_info['hidden_dim1'], model_info['hidden_dim2'], model_info['output_dim'])

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model


def _get_train_data_loader(batch_size, training_dir):
    """
    Gets training data in batches from the `train.csv` file
    
    :param batch_size: size of the batch
    :param training_dir: training directory
    
    :return: PyTorch DataLoader object
    """
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train_nn.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def RMSELoss(y_pred, y_test):
    """
    Computes the RMSE loss
    
    :param y_pred: prediction vector
    :param y_test: test vector
    
    :return: RMSE
    """
    return torch.sqrt(torch.mean((y_pred-y_test)**2))

def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    Training method called by the PyTorch training script.
    
    :param model: PyTorch model that we wish to train
    :param train_loader: PyTorch DataLoader to be used during training
    :param epochs: total number of epochs to train for
    :param criterion: loss function used for training
    :param optimizer: optimizer to use during training
    :param device: where the model and data should be loaded (gpu or cpu)
    :return:
    """
    
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode

        total_loss = 0

        for batch in train_loader:
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            y_pred = model(batch_x)
            
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--input_features', type=int, default=3, metavar='IN',
                        help='number of input features to model (default: 3)')
    parser.add_argument('--hidden_dim1', type=int, default=128, metavar='H',
                        help='hidden dim1 of model (default: 128)')
    parser.add_argument('--hidden_dim2', type=int, default=64, metavar='H',
                        help='hidden dim2 of model (default: 64)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='output dim of model (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    model = Regression(args.input_features, args.hidden_dim1, args.hidden_dim2, args.output_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = RMSELoss

    train(model, train_loader, args.epochs, criterion, optimizer, device)

    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim1': args.hidden_dim1,
            'hidden_dim2': args.hidden_dim2,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)
        
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
