import torch
import sys
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import load_data, evaluate, plot_confusion_matrix
from config import model_config as config


class LSTMClassifier(nn.Module):
    """docstring for LSTMClasifier"""

    def __init__(self, config):
        super().__init__()
        self.n_layers = config['n_layers']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.bidirectional = config['bidirectional']
        self.dropout = config['dropout'] if self.n_layers > 1 else 0

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
                           num_layers=2, droupot=self.dropout, bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = F.softmax()

    def forward(self, input_seq):
        # input_seq = [1, batch_size, input_size]
        rnn_output, (hidden, _) = self.rnn(input_seq)
        if self.bidirectional:  # Sum outputs from the two direcations
            rnn_output = rnn_output[:, :, :self.hidden_dim] + \
                rnn_output[:, :, self.hidden_dim:]
            class_scores = F.softmax(self.out(rnn_output[0]), dim=1)
            return class_scores


if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}

    device = 'cuda:{}'.format(
        config['gpu']) if torch.cuda.is_available() else 'cpu'

    model = LSTMClassifier(config)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_batches = load_data()
    test_pairs = load_data(test=True)

    best_acc = 0
    for epoch in range(config['n_epoch']):
        losses = []
        for batch in train_batches:
            inputs = batch[0].unsqueeze(0)
            targets = batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero_grad() function is used to clear the gradients of all parameters in a model, and it's often called before running the backward() function during training
            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs)
            predictions = predictions.to(device)

            loss = criterion(predictions, targets)
            # The backward() method in Pytorch is used to calculate the gradient during the backward pass in the neural network. If we do not call this backward() method then gradients are not calculated for the tensors. The gradient of a tensor is calculated for the one having requires_grad is set to True.
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # evaluate
        # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
        with torch.no_grad():
            inputs = test_pairs[0].unsqueeze()
            targets = test_pairs[1]

            inputs = inputs.to(device)
        targets = targets.to(device)

        # take argmax to get class id
        # Torch.argmax() method accepts a tensor and returns the indices of the maximum values of the input tensor across a specified dimension/axis.
        predictions = torch.argmax(model(inputs), dim=1)
        predictions = predictions.to(device)

        # evaluate on cpu
        targets = np.array(targets.cpu())
        predictions = np.array(predictions.cpu())

        # Get results
        # plot_confusion_matrix(targets, predictions,
        #                       classes=emotion_dict.keys())
        performance = evaluate(targets, predictions)
        if performance['acc'] > best_acc:
            best_acc = performance['acc']
            print(performance)
            # save model and results
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, 'runs/{}-best_model.pth'.format(config['model_code']))
        # A file with a . pth extension typically contains a serialized PyTorch state dictionary. A PyTorch state dictionary is a Python dictionary that contains the state of a PyTorch model, including the model's weights, biases, and other parameters.
            with open('results/{}-best_performance.pkl'.format(config['model_code']), 'wb') as f:
                pickle.dump(performance, f)
