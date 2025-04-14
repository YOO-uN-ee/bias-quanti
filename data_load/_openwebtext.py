from datasets import load_dataset

def load_openwebtext(cache_dir:str=None):
    if cache_dir:
        openwebtext = load_dataset("Skylion007/openwebtext", cache_dir=cache_dir, trust_remote_code=True)
    else:
        openwebtext = load_dataset("Skylion007/openwebtext", trust_remote_code=True)

    return openwebtext