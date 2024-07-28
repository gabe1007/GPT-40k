import torch 
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.bpe import BPE

BATCH_SIZE = 16 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 256 # what is the maximum context length for predictions?
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 1000
N_EMBED = 512
N_HEADS = 8
DROPOUT = 0.2
EVAL_ITERS = 200
N_LAYER = 6

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
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
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
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBED) # 512 x 512
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEADS) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED) # final layer norm
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE) # language model head

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

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

    foo = Block(N_EMBED, N_HEADS)
    foo.forward(new_emb)
    
    

