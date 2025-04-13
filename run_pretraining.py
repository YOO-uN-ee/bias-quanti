import time
import argparse

import data_load, pretrain

def main(model_name:str,
         datasets:list,
         cache_directory:str=None) -> None:
    
    model_name = 'FacebookAI/roberta-base'
    dataset = data_load.load_bookcorpus(cache_dir=cache_directory)

    pretrain.train_tokenizer(output_directory='/home/yaoyi/pyo00005/quanti_bias/_models/', output_name='_bc_roberta',
                             dataset=dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='quanti_bias')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--datasets', type=str) # convert to option list
    parser.add_argument('--cache_directory', type=str, default=None)
    args = parser.parse_args()

    main(model_name=args.model_name, 
         datasets=[args.datasets], 
         cache_directory=args.cache_directory)