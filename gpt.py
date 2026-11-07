import torch
torch.manual_seed(1337)

#hyperparameters
batch_size = 16
block_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

#train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split=='train_data' else val_data
    ix = torch.randint(len(d)-block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == '__main__':
    print(f'{len(text):,} chars, vocab {vocab_size}: {"".join(chars)!r}')
    assert decode(encode('hii there')) == 'hii there'
    xb, yb = get_batch('train')
    print(xb.shape, yb.shape)
    print(xb[0].tolist())
    print(yb[0].tolist())