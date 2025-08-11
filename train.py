from model.rnn_model import Model 
from utils.data_utils import load_data, get_data_loader, train_test_split
from tokenizers import Tokenizer 
import torch.nn as nn 
import torch 

EPOCHS = 10

input_vocab_size = Tokenizer.from_file("./trained_tokenizers/input_tokenizer.json").get_vocab_size()
output_vocab_size = Tokenizer.from_file("./trained_tokenizers/output-tokenizer.json").get_vocab_size() 


model = Model(input_vocab=input_vocab_size, output_vocab_size=output_vocab_size) 

data = load_data("./data/processed_data.pt") 

train_set, test_set = train_test_split(0.7, data) 

train_loader = get_data_loader(train_set) 
test_loader = get_data_loader(test_set)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.5)

for i in range(EPOCHS): 

    print(f"training loss : {model.train(train_loader, loss, optimizer)}") 

    print(f"test loss: {model.test(test_loader, loss)}")