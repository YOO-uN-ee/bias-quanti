from datasets import load_dataset

def load_wiki(cache_dir:str=None):
    if cache_dir:
        wiki = load_dataset("wikipedia", "20220301.en", cache_dir=cache_dir, trust_remote_code=True)
    else:
        wiki = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)

    return wiki