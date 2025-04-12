from datasets import load_dataset, concatenate_datasets, load_from_disk
import time

cache_dir = "/home/yaoyi/pyo00005/.cache"

# book corpus
start_time = time.time()
bookcorpus = load_dataset("bookcorpus", cache_dir=cache_dir, trust_remote_code=True)
print(bookcorpus, time.time() - start_time)

start_time = time.time()
wiki = load_dataset("wikipedia", "20220301.en", cache_dir=cache_dir, trust_remote_code=True)
print(wiki, time.time() - start_time)



start_time = time.time()
clue = load_dataset('irds/clueweb09', 'docs', cache_dir=cache_dir, trust_remote_code=True)
print(clue, time.time() - start_time)

# english wikipedia
# wiki = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir)
# wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

# # # concatenation
# concat = concatenate_datasets([bookcorpus, wiki])

# concat.push_to_hub("JackBAI/bert_pretrain_datasets")