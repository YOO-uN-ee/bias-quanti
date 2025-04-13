from datasets import load_dataset

def load_bookcorpus(cache_dir:str=None):
    if cache_dir:
        bookcorpus = load_dataset("bookcorpus", cache_dir=cache_dir, trust_remote_code=True)
    else:
        bookcorpus = load_dataset("bookcorpus", trust_remote_code=True)

    return bookcorpus