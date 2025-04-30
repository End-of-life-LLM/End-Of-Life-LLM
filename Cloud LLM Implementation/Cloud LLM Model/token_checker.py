import tiktoken 

class Token_Checker:
    def __init__(self):
        pass 

    def count_tokens(text: str, model: str = "text-embedding-3-small") -> int: 
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)

