from model.rnn_model import Model 
from utils.data_utils import load_data, get_data_loader, train_test_split
from tokenizers import Tokenizer 
from collections import defaultdict
import seaborn as sns 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch 
import logging 

torch.random.manual_seed(42)

EPOCHS = 10
EMBEDDING_SIZE = 8
HIDDEN_SIZE = 8
NUM_LSTM_LAYERS = 2
BATCH_SIZE = 64

logging.basicConfig(level = logging.INFO) 

def plot_history(history: defaultdict): 

    for key in history.keys(): 

        sns.lineplot(history, x = range(1, len(history[key])+1), y = key, markers=True, marker = 'o')
        plt.xlabel("Epochs") 
        plt.xticks(range(1, len(history[key])+1))
        plt.ylabel(f"{key}")
        plt.grid(True) 
        plt.savefig(f"./plots/{key}.png")
        plt.close() 

if __name__ == '__main__': 

    logging.info("initializing training script") 

    input_vocab_size = Tokenizer.from_file("./trained_tokenizers/input_tokenizer.json").get_vocab_size()
    output_vocab_size = Tokenizer.from_file("./trained_tokenizers/output-tokenizer.json").get_vocab_size() 

    model = Model(input_vocab=input_vocab_size, 
                output_vocab_size=output_vocab_size,
                embedding_size = EMBEDDING_SIZE, 
                hidden_size = HIDDEN_SIZE,
                num_lstm_layers=NUM_LSTM_LAYERS) 

    logging.info("loading data for training")

    data = load_data("./data/processed_data.pt") 

    train_set, test_set = train_test_split(0.7, data) 

    train_loader = get_data_loader(train_set, batch_size = BATCH_SIZE) 
    test_loader = get_data_loader(test_set, batch_size=BATCH_SIZE)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

    torch.compile(model) 

    logging.info("Starting the training loop...")

    train_history = defaultdict(list) 

    for i in range(1, EPOCHS+1): 

        train_loss, accuracy = model.train(train_loader, loss, optimizer) 
        test_loss, test_accuracy = model.test(test_loader, loss)

        train_history['train_loss'].append(train_loss) 
        train_history['train_accuracy'].append(accuracy) 
        train_history['test_loss'].append(test_loss) 
        train_history['test_accuracy'].append(test_accuracy) 

        logging.info(f"epoch {i} training done...")

        logging.info(f"training loss: {train_loss} | training accuracy: {accuracy}") 

        logging.info(f"test loss: {test_loss} | test accuracy: {test_accuracy}")

    plot_history(train_history) 

    torch.save(model.state_dict(), "./trained_model/model.pth")
    