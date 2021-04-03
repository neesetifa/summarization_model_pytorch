import torch
import pandas as pd
from dataset import MyDataset, my_collate_fn
from models.model_lstm import LSTM_Model
from loss import SimpleCCELoss
from utils import load_vocab, evaluate
import config


def main():
    data = pd.read_csv(config.train_dataset_file)
    train_data = data.iloc[20:]
    validation_data = data.iloc[:20]
    
    chars, id2char, char2id = load_vocab(config.vocabulary_file, config.train_dataset_file)
    
    vocab_size = len(chars)  # 5830, Does NOT include 4 special chars
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device check. You are using:", device)
    
    train_dataset = MyDataset(data = train_data,
                              char2id = char2id)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                   batch_size = config.batch_size, 
                                   shuffle = True,
                                   collate_fn = my_collate_fn)

    validation_dataset = MyDataset(data = validation_data,
                                   char2id = char2id,
                                   convert_to_ids = False)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                   batch_size = 1, 
                                   shuffle = False)
    
    model = LSTM_Model(vocab_size = vocab_size,
                       embedding_size = config.embedding_size,
                       hidden_size = config.hidden_size,
                       device = device).to(device)

    criterion = SimpleCCELoss(device = device)
    optimizer = torch.optim.Adam(params = model.parameters())
    epochs = config.epochs

    best_training_loss = float('inf')
    for epoch in range(epochs):
        # training part
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(train_data_loader):
            x, y, len_x, len_y = data  
            x, y, len_x, len_y = x.to(device), y.to(device), len_x, len_y
            
            model.train()
            output = model(x, y, len_x, len_y)
            loss = criterion(output, y)
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step() 

            running_loss += loss.item()
            epoch_loss += loss.item() 
            if i!=0 and i%50 == 0:
                print('Epoch {}/{}, Step {} - loss: {}'.format(epoch+1, epochs, i, running_loss/50))
                running_loss = 0.0
            
        torch.save(model.state_dict(), 'saved_weight/saved_model_on_epoch_{}.pth'.format(epoch+1))
        epoch_loss /= len(train_data_loader)
        if epoch_loss < best_training_loss:
            best_training_loss = epoch_loss
            torch.save(model.state_dict(), 'saved_weight/best_model_training.pth')
        print('Training loss: {}, Best training loss: {}'.format(epoch_loss, best_training_loss))

        
        # validation part
        evaluate(validation_data_loader, model, device, char2id, id2char)
        
            
if __name__ == '__main__':
    main()