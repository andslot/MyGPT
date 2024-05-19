import tiktoken

# class Embedding():
#     def __init__(self):
#         self.encoder = tiktoken.encoding_for_model("gpt-2")
#         self.vocab_size = self.encoder.n_vocab
    
#     def encode(self, input: str = None):
#         assert input is not None, "Require some input string."
#         return self.encoder.encode(input).tolist()
    
#     def decode(self, input: list[int] = None) -> list:
#         assert input is not None, "Require encoded list of ints"
#         return self.encoder.decode(input)

class Embedding():
    def __init__(self):
        with open('downloads/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join([itos[i] for i in l])