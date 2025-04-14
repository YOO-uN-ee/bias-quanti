import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def train_tokenizer(output_directory:str, output_name:str,
                    dataset=None,
                    input_directory:str=None, input_format:str='txt',
                    vocab_size:int=50000, min_frequency:int=2,
                    special_tokens:list=["<s>","<pad>","</s>","<unk>","<mask>"]) -> None:
    """
    Train BPE tokenizer
    Creates vocab.json which lists most frequent tokens ranked by frequency and merges.txt
    """
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Load training data
    if input_directory:
        paths = [str(x) for x in Path(input_directory).glob(f"**/*.{input_format}")]
        tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    if dataset:
        training_corpus = dataset['train']['text'] # might need to find out what the text portion is for each
        tokenizer.train_from_iterator(training_corpus, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    try:
        os.makedir(output_directory)
    except:
        pass

    # Save tokenizer
    tokenizer.save_model(output_directory, output_name)

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    tokenizer.save(os.path.join(output_directory, f"{output_name}.json"))

# def test_tokenizer(tokenizer_directory:str, tokenizer_name:str):
#     tokenizer = ByteLevelBPETokenizer(
#         f"./models/EsperBERTo-small/vocab.json",
#         f"./models/EsperBERTo-small/merges.txt",
#     )
#     tokenizer._tokenizer.post_processor = BertProcessing(
#         ("</s>", tokenizer.token_to_id("</s>")),
#         ("<s>", tokenizer.token_to_id("<s>")),
#     )
#     tokenizer.enable_truncation(max_length=512)

#     print(
#         tokenizer.encode("Mi estas Julien.")
#     )