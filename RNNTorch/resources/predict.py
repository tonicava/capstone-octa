import os
import numpy as np
import torch
from six import BytesIO

from model import Regression


NP_CONTENT_TYPE = 'application/x-npy'


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


def input_fn(serialized_input_data, content_type):
    """
    Takes the request data and deserializes the data into an object for prediction
    
    :param serialized_input_data: data to be deserialized
    :param content_type: the MIME type of the data in serialized_input_data
    :return: deserialized object
    """
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    """
    Takes the result of the prediction and serializes it according to the response content type
    
    :param prediction_output: data to be serialized
    :param accept: the MIME type of the data to be serialized
    :return: serialized object
    """    
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    """
    Takes the deserialized request object and performs inference against the loaded model
    
    :param input_data: request data
    :param model: loaded model
    :return: a NumPy array containing predictions to be returned to the client
    """    
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    model.eval()

    out = model(data)
    out_np = out.cpu().detach().numpy()

    return out_np