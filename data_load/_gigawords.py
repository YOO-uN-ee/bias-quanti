from datasets import load_dataset

def load_gigawords(cache_dir:str=None):
    if cache_dir:
        gigawords = load_dataset("Harvard/gigaword", cache_dir=cache_dir, trust_remote_code=True)
    else:
        gigawords = load_dataset("Harvard/gigaword", trust_remote_code=True)

    return gigawords