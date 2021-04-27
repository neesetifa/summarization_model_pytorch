import torch
from torch import nn
import torch.nn.functional as F
import math

class ScaleShift(nn.Module):
   # scale and shift layer 
   def __init__(self, 
                input_shape, 
                init_value, # 1e-3
                device):
       
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value] * input_shape).to(device))
       self.bias = nn.Parameter(torch.FloatTensor([init_value] * input_shape).to(device))

   def forward(self, inputs):
       return torch.exp(self.scale) * inputs + self.bias


class Attention(nn.Module):
    def __init__(self,
                 num_attention_heads,
                 hidden_size):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # usually hidden_size = all_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
    
    def mask(self, x, mask = None, mode = 'mul'):
        if mask == None:
            print('Hint: mask is None, original x is returned.')
            return x
        
        # x shape = [batch_size, seq_len, any_size]
        # mask shape = [batch_size, seq_len, 1]
        if len(x.shape)>3: # if x shape = [batch_size, seq_len, size_1, size_2]
            original_shape = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1) # reshape to [batch_size, seq_len, size_1 * size_2]
        
        # [batch_size, seq_len, 1] -> [batch_size, seq_len, any_size]
        mask = mask.repeat(1, 1, x.shape[-1])
        
        if mode == 'mul':
            x = x * mask
        elif mode == 'add':
            x = x - (1 - mask) * 1e10
        else:
            raise ValueError('Got mode {}, Only accept mode to be "add" or "mul"'.format(mode))
        
        if x.shape != original_shape: # view back to [batch_size, seq_len, size_1, size_2]
            x = x.reshape(*original_shape)
        return x
        
    def do_transpose(self, x):
       new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
       # [batch_size, seq_len, out_dim] -> [batch_size, seq_len, num_heads, head_size]
       x = x.reshape(*new_shape)
       # [batch_size, seq_len, num_heads, head_size] -> [batch_size, num_heads, seq_len, head_size]
       return x.permute(0, 2, 1, 3)
        
    def forward(self, qx, kx, vx, v_mask = None, q_mask = None):
        # we assume kx == vx, so seq_len_k = seq_len_v
        # if mask is passed, size should be [batch_size, seq_len, 1]
        
        # [batch_size, seq_len_q/kv, hidden_size] -> [batch_size, seq_len_q/kv, all_head_size]
        qw = self.query(qx)
        kw = self.key(kx)
        vw = self.value(vx)
        
        # [batch_size, num_heads, seq_len_q/kv, head_size]
        qw = self.do_transpose(qw)
        kw = self.do_transpose(kw)
        vw = self.do_transpose(vw)
        
        # [batch_size, num_heads, seq_len_q, seq_len_kv]
        attention_scores = torch.matmul(qw, kw.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if v_mask is not None:
            # [batch_size, seq_len_kv, seq_len_q, num_heads]
            attention_scores = attention_scores.permute(0, 3, 2, 1)
            attention_scores = self.mask(attention_scores, v_mask, mode = 'add')
            # [batch_size, num_heads, seq_len_q, seq_len_kv]
            attention_scores = attention_scores.permute(0, 3, 2, 1)
        
        # do softmax on seq_len_k, because we need to find weights of key sequence
        attention_probs = nn.Softmax(dim = -1)(attention_scores)
        
        # [batch_size, num_heads, seq_len_q, seq_len_kv] -> [batch_size, num_heads, seq_len_q, head_size]
        output = torch.matmul(attention_probs, vw)
        # [batch_size, seq_len_q, num_heads, head_size]
        output = output.permute(0, 2, 1, 3)
        
        # [batch_size, seq_len_q, hidden_size]
        output_shape = output.size()[:-2] + (self.all_head_size, )
        output = output.reshape(*output_shape)
        
        if q_mask is not None:
            output = self.mask(output, q_mask, mode = 'mul')
        
        # [batch_size, seq_len_q, hidden_size]
        return output
        
        
        
class LSTM_Model(nn.Module):
    def __init__(self, 
                 vocab_size, # 5830, total should be 5830 + 4 special symbols = 5834
                 embedding_size, # 256
                 hidden_size,    # 256
                 device
                 ):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.device = device
        
        self.embedding = nn.Embedding(num_embeddings = vocab_size + 4, 
                                      embedding_dim = embedding_size)
        
        self.lstm_x = nn.LSTM(input_size = embedding_size, 
                            hidden_size = hidden_size // 2,
                            num_layers = 2,
                            batch_first = True,
                            bidirectional = True)
        
        self.lstm_y = nn.LSTM(input_size = embedding_size, 
                            hidden_size = hidden_size,
                            num_layers = 2,
                            batch_first = True)
        
        self.scale_shift = ScaleShift(input_shape = vocab_size + 4, init_value = 1e-3, device = device)
        
        self.layer_norm_x = nn.LayerNorm(normalized_shape = hidden_size)
        self.layer_norm_y = nn.LayerNorm(normalized_shape = hidden_size)
        self.linear_1 = nn.Linear(in_features = 2 * hidden_size, out_features = embedding_size)
        self.linear_2 = nn.Linear(in_features = embedding_size, out_features = vocab_size + 4)
        
        self.attention = Attention(num_attention_heads = 8,
                                   hidden_size = hidden_size)
        
        self.activation = nn.ReLU(inplace = True)
        self.softmax = nn.LogSoftmax(dim = -1)
        
    
    def to_one_hot(self, x, mask):
        # [batch_size, seq_len] -> [batch_size, seq_len, vocab_size+4]
        x = F.one_hot(x, num_classes = self.vocab_size + 4)
        # [batch_size, seq_len, 1] -> [batch_size, seq_len, vocab_size+4]
        mask = mask.repeat(1, 1, self.vocab_size + 4)
        # [batch_size, 1, vocab_size+4]
        x = torch.sum(x * mask, dim = 1, keepdim = True) 
        x = torch.where(x >= 1., torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
        return x
        
    
    def forward(self, x, y, len_x, len_y):
        # x is input text, shape = [batch_size, seq_len_x]
        # y is label summarization, shape = [batch_size, seq_len_y]
       
        # mask = [batch_size, seq_len, 1]
        x_mask = torch.where(x > 1., torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device)).unsqueeze(-1)
        
        # [batch_size, 1, vocab_size+4]
        x_one_hot = self.to_one_hot(x, x_mask)
        x_prior = self.scale_shift(x_one_hot)
        
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_size]
        x = self.embedding(x)
        y = self.embedding(y)
        
        # let torch do sort, so we set enforce_sorted = False
        x = nn.utils.rnn.pack_padded_sequence(x, lengths = len_x, batch_first = True, enforce_sorted = False)
        x, (h_x, c_x) = self.lstm_x(x)
        # x shape [batch_size, seq_len_x, hidden_size]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = self.layer_norm_x(x)

        y = nn.utils.rnn.pack_padded_sequence(y, lengths = len_y, batch_first = True, enforce_sorted = False)
        y, (h_y, c_y) = self.lstm_y(y)
        # y shape [batch_size, seq_len_y, hidden_size]
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first = True)
        y = self.layer_norm_y(y)
        
        # xy shape [batch_size, seq_len_y, hidden_size]
        xy = self.attention(qx = y,
                            kx = x,
                            vx = x,
                            v_mask = x_mask)

        # xy shape [batch_size, seq_len_y, 2*hidden_size]
        xy = torch.cat((y, xy), dim = -1)
        # xy shape [batch_size, seq_len_y, embedding_size]
        xy = self.linear_1(xy)
        xy = self.activation(xy)
        # xy shape [batch_size, seq_len_y, vocab_size + 4]
        xy = self.linear_2(xy)
        # xy shape [batch_size, seq_len_y, vocab_size + 4]
        xy = (xy + x_prior) / 2.
        xy = self.softmax(xy)
        # xy shape [batch_size, seq_len_y, vocab_size + 4]
        return xy
