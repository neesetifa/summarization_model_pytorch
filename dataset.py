import torch
from torch import nn
from utils import str2id


def my_collate_fn(data):
    len_text, len_summary = [], []
    text, summary = [], []
    for t, s in data:
        len_text.append(len(t))
        text.append(t)
        len_summary.append(len(s))
        summary.append(s)
        
    text = nn.utils.rnn.pad_sequence(text, batch_first = True, padding_value = 0)
    summary  = nn.utils.rnn.pad_sequence(summary, batch_first = True, padding_value = 0)
    return text, summary, torch.tensor(len_text), torch.tensor(len_summary)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, char2id, convert_to_ids = True):
        self.data = data
        self.char2id = char2id
        self.convert_to_ids = convert_to_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text, summary = self.data.values[index]
        if not self.convert_to_ids:
            return text, summary  # pytorch will return (text,), (summary,)  two tuples
        
        text = str2id(text, self.char2id)
        summary = str2id(summary, self.char2id, start_end = True)
        return torch.tensor(text), torch.tensor(summary)
             
