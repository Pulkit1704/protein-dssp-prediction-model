import torch 
import torch.nn as nn 
from sklearn.metrics import accuracy_score 

class Model(nn.Module): 

    def __init__(self, input_vocab, 
                 output_vocab_size, 
                 embedding_size=32, 
                 hidden_size=16, 
                 num_lstm_layers = 2):

        super().__init__() 

        self.embedding = nn.Embedding(input_vocab, embedding_size) 

        self.rnn = nn.LSTM(embedding_size, 
                           hidden_size,
                           num_layers=num_lstm_layers, 
                           batch_first=True,
                           bidirectional = True) 

        self.dense = nn.Sequential(
                        nn.Linear(int(hidden_size*2), output_vocab_size),
                        nn.ReLU()
                        )

        self.activation = nn.Softmax(dim=-1) 


    def forward(self, x): 
        x = self.embedding(x) 

        x, _ = self.rnn(x) 

        x = self.dense(x) 

        return x 
    
    def predict(self, x): 

        outputs = self.forward(x) 
        
        return self.activation(outputs)
    
    def train(self, train_batch, loss_fn, optimizer): 

        train_loss = 0
        train_score = 0
        train_size = 0

        for i, data in enumerate(train_batch): 

            optimizer.zero_grad()

            inputs, targets = data 

            predictions = self.forward(inputs) 

            loss = loss_fn(predictions.view(-1, predictions.shape[-1]), 
                           targets.view(-1)) 

            train_loss += loss.item() 

            train_score += accuracy_score(targets.cpu().view(-1),
                                         torch.argmax(self.predict(inputs).cpu(), dim = -1).view(-1), normalize = False) 
            train_size += targets.shape[0] * targets.shape[1]

            loss.backward() 

            optimizer.step()

        train_accuracy = train_score/train_size

        return train_loss, train_accuracy
    
    def test(self, test_batch, loss_func): 

        test_loss = 0 
        test_score = 0
        test_size = 0

        for i, data in enumerate(test_batch): 

            inputs, targets = data 

            predictions = self.forward(inputs) 

            loss = loss_func(predictions.view(-1, predictions.shape[-1]), 
                             targets.view(-1)) 

            test_loss += loss.item() 

            test_score += accuracy_score(targets.cpu().view(-1),
                                         torch.argmax(self.predict(inputs).cpu(), dim = -1).view(-1), normalize = False) 
            test_size += targets.shape[0] * targets.shape[1]
        
        test_accuracy = test_score/test_size
        
        return test_loss, test_accuracy