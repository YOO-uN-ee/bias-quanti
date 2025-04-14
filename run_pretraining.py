import time
import argparse
import os

import torch
import data_load, pretrain
# from datasets import Dataset, DatasetDict

torch.cuda.empty_cache()

def main(model_name:str,
         dataset_name:list,
         train_batch_size:int,
         eval_batch_size:int,
         gradient_accumulation:int,
         cache_directory:str=None) -> None:
    
   model_name = 'FacebookAI/roberta-base'

   if dataset_name == 'bc':
      dataset = data_load.load_bookcorpus(cache_dir=cache_directory)
   elif dataset_name == 'wiki':
      dataset = data_load.load_wiki(cache_dir=cache_directory)
   else:
      raise ValueError('Not acceptable dataset type')

   #  sampled_train = dataset['train'].select(range(100)) 
   #  dataset = DatasetDict()
   #  dataset['train'] = sampled_train

   raw_model_name = model_name.split('/')[1]
   tokenizer_name = 'roberta'

   if os.path.exists(f'/home/yaoyi/pyo00005/quanti_bias/_models/_{dataset_name}_{tokenizer_name}'):
      pass
   else:
      pretrain.train_tokenizer(output_directory=f'/home/yaoyi/pyo00005/quanti_bias/_models/_{dataset_name}_{raw_model_name}/', output_name=f'_{dataset_name}_{tokenizer_name}',
                              dataset=dataset)
   pretrain.run_train(base_model_name=model_name,
                       trained_tokenizer=f'./_models/_{dataset_name}_{raw_model_name}',
                       trained_model=f'/home/yaoyi/pyo00005/quanti_bias/_models/_{dataset_name}_{raw_model_name}',
                       dataset=dataset,
                       per_device_train_batch_size=train_batch_size,
                       per_device_eval_batch_size=eval_batch_size,
                       gradient_accumulation_steps=gradient_accumulation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='quanti_bias')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--datasets', type=str) # convert to option list
    parser.add_argument('--cache_directory', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    args = parser.parse_args()

    main(model_name=args.model_name, 
         dataset_name=args.datasets, 
         cache_directory=args.cache_directory,
         train_batch_size=args.train_batch_size,
         eval_batch_size=args.eval_batch_size,
         gradient_accumulation=args.gradient_accumulation)