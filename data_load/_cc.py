from datasets import load_dataset, concatenate_datasets

def load_ccnews(cache_dir:str=None,
                start_year:int=2016,
                end_year:int=None,
                language:str='en'):
    
    list_datasets = []
    list_years = [start_year]

    if end_year:
        list_years = list(range(start_year, end_year+1))

    for year in list_years:
        if cache_dir:
            list_datasets.append(load_dataset("stanford-oval/ccnews", name=str(year), cache_dir=cache_dir, trust_remote_code=True))
        else:
            list_datasets.append(load_dataset("stanford-oval/ccnews", name=str(year), trust_remote_code=True))

    ccnews = concatenate_datasets(list_datasets)

    return ccnews

def load_cc():
    return 0

def load_ccstories(cache_dir:str=None,):
    if cache_dir:
        ccstories = load_dataset("spacemanidol/cc-stories", cache_dir=cache_dir, trust_remote_code=True)
    else:
        ccstories = load_dataset("spacemanidol/cc-stories", trust_remote_code=True)

    return ccstories