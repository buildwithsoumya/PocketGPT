import torch
from gpt import GPTLanguageModel, encode, decode, stoi, device

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('pocketgpt.pt', map_location=device))
model.eval()
print("PocketGPT ready. Type a prompt (empty line to quit).")
while True:
    prompt = input('> ')
    if not prompt:
        break
    prompt = prompt.replace('\\n', '\n')   
    unknown = set(prompt)-set(stoi)
    if unknown:
        print(f"Unknown characters: {''.join(sorted(unknown))}")
        continue
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=300, temperature=0.8)
    print(decode(out[0].tolist()))