import torch
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import model_config as config
from utils import load_data, evaluate, load_word_embeddings, plot_confusion_matrix


class LSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""

    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.n_layers = config['n_layers']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.bidirectional = config['bidirectional']

        self.embedding = nn.Embedding.from_pretrained(
            load_word_embeddings(), freeze=False
        )

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout, bidirectional=self.bidirectional)

        self.n_directions = 2 if self.bidirectional else 1
        self.out = nn.Linear(self.n_directions *
                             self.hidden_dim, self.output_dim)
        self.softmax = F.softmax

    def forward(self, input_seq, input_lengths):
        max_seq_len, bs = input_seq.size()
        embedded = self.embedding(input_seq)

        # Below code takes the output of an RNN module, selects the last and first time steps of the output tensor, and concatenates specific portions of these time steps to create a new tensor.

        rnn_output, (hidden, _) = self.rnn(embedded)
        rnn_output = torch.cat(
            (rnn_output[-1, :, :self.hidden_dim], rnn_output[0, :, self.hidden_dim:]), dim=1)

        # The above code is useful where bidirectional information is important, such as named entity recognition or machine translation. By considering both the initial and final states of the RNN, the model can capture information from both the beginning and end of the input sequence.

        # Sum hidden states
        class_scores = F.softmax(self.out(rnn_output), dim=1)
        return class_scores


if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}

    device = 'cuda:{}'.format(
        config['gpu']) if torch.cuda.is_available() else 'cpu'

    model = LSTMClassifier(config)
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_batches = load_data()
    test_batches = load_data(test=True)

    best_acc = 0
    for epoch in range(config['n_epochs']):
        losses = []
        for batch in train_batches:
            inputs, input_length, targets = batch
            inputs = inputs.to(device)
            input_length = input_length.to(device)
            targets = targets.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs, input_length)
            predictions = predictions.to(device)

            loss = criterion(predictions, targets)

            # The backward pass, also known as backpropagation, is a key step in this process. It calculates the gradients of the loss with respect to the model's parameters, which indicate the direction and magnitude of the parameter updates needed to minimize the loss.
            loss.backward()

            optimizer.step()

            losses.append(loss.item())
        # Evaluate
        with torch.no_grad():
            inputs, lengths, targets = test_batches

            inputs = inputs.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            # take argmax to get class id
            predictions = torch.argmax(model(inputs, lengths), dim=1)
            predictions = predictions.to(device)

            # Evaluate on cpu
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())

            # Get results

            performance = evaluate(targets, predictions)

            if performance['acc'] > best_acc:
                best_acc = performance['acc']
                # Save model and results
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, 'runs/{}-best_model.pth'.format(config['model_code']))

                with open('results/{}-best_performace.pkl'.format(
                        config['model_code']), 'wb') as f:
                    pickle.dump(performance, f)
