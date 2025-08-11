import torch.nn as nn 

class Model(nn.Module): 

    def __init__(self, input_vocab, output_vocab_size, embedding_size=32, hidden_size=16):

        super().__init__() 

        self.embedding = nn.Embedding(input_vocab, embedding_size) 

        self.dropout  = nn.Dropout(0.5) 

        self.rnn = nn.LSTM(embedding_size, hidden_size,num_layers = 5, batch_first=True, dropout= 0.5) 

        self.dense = nn.Linear(hidden_size, output_vocab_size) 

        self.activation = nn.Softmax() 


    def forward(self, x): 
        x = self.embedding(x) 

        x, _ = self.rnn(self.dropout(x)) 

        x = self.dense(x) 

        return x 
    
    def train(self, train_batch, loss_fn, optimizer): 

        train_loss = 0

        for i, data in enumerate(train_batch): 

            optimizer.zero_grad()

            inputs, targets = data 

            predictions = self.forward(inputs) 

            loss = loss_fn(predictions.view(-1, predictions.shape[-1]), targets.view(-1)) 

            train_loss += loss.item() 

            loss.backward() 

            optimizer.step()

        return train_loss 
    
    def test(self, test_batch, loss_func): 

        test_loss = 0 

        for i, data in enumerate(test_batch): 

            inputs, targets = data 

            predictions = self.forward(inputs) 

            loss = loss_func(predictions.view(-1, predictions.shape[-1]), targets.view(-1)) 

            test_loss += loss.item() 

        
        return test_loss 