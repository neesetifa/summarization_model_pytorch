import torch
import pandas as pd
from dataset import MyDataset
from models.model_lstm import LSTM_Model
from utils import load_vocab, evaluate
import config


def main():
    test_data = pd.read_csv(config.test_dataset_file)
    
    chars, id2char, char2id = load_vocab(config.vocabulary_file, '.')
    vocab_size = len(chars)  # 5830, Does NOT include 4 special chars
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Device check. You are using:", device)

    # 2000 sentences
    test_dataset = MyDataset(data = test_data,
                                   char2id = char2id,
                                   convert_to_ids = False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                   batch_size = 1, 
                                   shuffle = False)

    model = LSTM_Model(vocab_size = vocab_size,
                       embedding_size = config.embedding_size,
                       hidden_size = config.hidden_size,
                       device = device).to(device)

    # For CPU
    #model.load_state_dict(torch.load('saved_weight/saved_model_on_epoch_74.pth', map_location=torch.device('cpu') ))
    # For GPU
    model.load_state_dict(torch.load('saved_weight/saved_model_on_epoch_74.pth')
    evaluate(test_data_loader, model, device, char2id, id2char)

if __name__ == '__main__':
    main()
