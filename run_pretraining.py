import time
import argparse

import data_load

def main(model_name:str,
         datasets:list) -> None:
    print(data_load.load_bookcorpus())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='quanti_bias')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--datasets', type=str) # convert to option list
    args = parser.parse_args()

    main(args.model_name, [args.datasets])