import torch 
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.bpe import BPE

BATCH_SIZE = 16 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 256 # what is the maximum context length for predictions?
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 1000
N_EMBED = 512
DROPOUT = 0.2

def convert_to_tensor(text):
    """
    Converts a text to a tensor.

    Args:
        text (str): The text to be converted.

    Returns:
        torch.Tensor: A tensor representing the text.
    """
    bpe = BPE()
    encoded_data = bpe.encode(text)
    data = torch.tensor(encoded_data, dtype=torch.long)
    return data

def split(tensors):
    """
    Splits a list of tensors into training and testing sets.

    Parameters:
        tensors (list): A list of tensors to be split.

    Returns:
        tuple: A tuple containing two lists - the training set and the testing set.
    """
    limit = int(0.9 * len(tensors))
    train = tensors[:limit]
    test = tensors[limit:]

    return train, test

def get_batch(split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    """
    Get a batch of data for training or testing.
    """
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_batch = []
    y_batch = []
    
    for i in ix:
        x_batch.append(data[i:i+block_size])
        y_batch.append(data[i+1:i+1+block_size])
    
    x = torch.stack(x_batch)
    y = torch.stack(y_batch)
    x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False) # head_size x N_EMBED
        self.query = nn.Linear(N_EMBED, head_size, bias=False) # head_size x N_EMBED
        self.value = nn.Linear(N_EMBED, head_size, bias=False) # head_size x N_EMBED
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # input of size (B, T, C)
        # output of size (B, T, head_size)
        B,T,C = x.shape
        #  x = B,T,C = 16, 256, 512,  self.key = 64 x 512, linear performs xAT+b, 
        # since nn.Linear inverts the order we have to transpose self.key to get 16 x 256 x 512 @ 512 x 64
        k = self.key(x)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T) dim=-1 means that the softmax function is applied along the last dimension of the tensor.
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


if __name__ == '__main__':
    with open('./Know_no_fear.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    text_ = convert_to_tensor(text[:1000])
    train, test = split(text_)
    X, y = get_batch('train')

    B, T = X.size()
    embeding = nn.Embedding(VOCAB_SIZE, N_EMBED)
    pos_embed = nn.Embedding(BLOCK_SIZE, N_EMBED)
    
    emb = embeding(X) # B,T,C
    emb_pos = pos_embed(torch.arange(T, device=DEVICE)) # T,C

    new_emb = emb + emb_pos # B,T,C + T,C

    head = Head(64)
    head.forward(new_emb) # B,T,C
    
    
    

