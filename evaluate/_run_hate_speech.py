import os
import time
import logging

from typing import Union, List, Dict

import numpy as np
import polars as pl
from tabulate import tabulate

import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Initiating tokenizer and language model that will be used for the discriminative model
tokenizer = AutoTokenizer.from_pretrained(params['MODEL_NAME'])

# Loading accuracy evaluation model that will be used in the training process
accuracy = evaluate.load('accuracy')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

def data_preprocessing(df_data:pl.DataFrame, 
                       col_truth:str=None,
                       col_split:str=None,
                       bool_testing:bool=False):
    if isinstance(df_data, pl.DataFrame):
        labels = list(set(df_data[col_truth]).to_list())
        id2label = {i: label.strip() for i, label in enumerate(labels)}
        label2id = {label.strip(): str(i) for i, label in id2label.items()}

        pl_text_data = pl_text_data.with_columns(
            label = pl.col(col_truth).replace(label2id, default=str(len(labels))).cast(pl.Int64)
        )

        # Split dataset into train, val, test
        if col_split:
            list_pl_data_splitted = pl_text_data.partition_by(
                col_split
            )

            pl_train = list_pl_data_splitted[0].drop(['train_val_test', 'order_idx']).to_pandas()
            pl_valid = list_pl_data_splitted[1].drop(['train_val_test', 'order_idx']).to_pandas()
            pl_test = list_pl_data_splitted[2].drop(['train_val_test', 'order_idx']).to_pandas()

        else:
            pass
            
        if bool_testing:
            pl_test = pl_test.drop('label', axis=1)

        data_ds = DatasetDict()
        data_ds['train'] = Dataset.from_pandas(pl_train)
        data_ds['validation'] = Dataset.from_pandas(pl_valid)
        data_ds['test'] = Dataset.from_pandas(pl_test)
    
    else: # Already in dataset format
        pass

    tokenized_data = data_ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_data, id2label, data_collator

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

def train_discriminative(tokenized_author_data, id2label:dict, data_collator):
    """
    Trains the model using the built in Trainer class from Huggingface Transformers

    : param: tokenized_author_data = DataDict with tokenized train, validation, and test set
    : param: id2label = dictionary that provides the mapping from numeric id to the author labeling
    : param: data_collator = DataCollator that does the batching; required as part of the trainer variable

    : return: trainer = trained model
    """
    # Training arguments
    # Reference: https://huggingface.co/transformers/v4.4.2/main_classes/trainer.html#trainingarguments
    logging.info('Training started')
    start_time = time.time()

    model = AutoModelForSequenceClassification.from_pretrained(
        params['MODEL_NAME'], num_labels=len(id2label)
    )

    training_args = TrainingArguments(
        output_dir='authorship_attribution_model',
        learning_rate=float(params['LEARNING_RATE']),
        per_device_train_batch_size=int(params['BATCH_SIZE']),
        per_device_eval_batch_size=int(params['BATCH_SIZE']),
        num_train_epochs=int(params['EPOCHS']),
        weight_decay=float(params['WEIGHT_DECAY']),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_author_data['train'],
        eval_dataset=tokenized_author_data['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    logging.info(f'Training ended - Run Time: {time.time() - start_time}')

    return trainer

def test_discriminative(tokenized_author_data, trainer, id2label:dict, bool_testing:bool):
    """
    Tests the model on the t
    Reference: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.predict
    
    : param: tokenized_author_data = 
    : param: trainer = 
    : param: id2label = dictionary consisting of the numeric id to author label mapping
    : param: bool_testing = boolean value indicating whether a testing file was provided (True) or not (False)
    """
    logging.info('Testing started')
    start_time = time.time()

    # Creates a output directory of store the development set result, predictions, and incorrect predictions
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    predictions, label_ids, metrics = trainer.predict(tokenized_author_data['test'])
    predicted_label = np.argmax(predictions, axis=1)
    test_text = tokenized_author_data['test']['text']

    if not bool_testing:
        test_label_ground_truth = tokenized_author_data['test']['label']
        
        compiled_data = pl.DataFrame(
            {
                "text":test_text,
                "author":test_label_ground_truth,
                "prediction":predicted_label
            }
        ).with_columns(
            pl.col('author').replace(id2label),
            pl.col('prediction').replace(id2label)
        )

        compiled_data.write_csv('./outputs/prediction_discriminative.csv',
                                separator=',')
        
        pl_incorrect_prediction = compiled_data.filter(
            pl.col('author') != pl.col('prediction')
        ).select(
            pl.col(['text', 'author', 'prediction'])
        )  
        pl_incorrect_prediction.write_csv('./outputs/incorrect_prediction_discriminative.csv',
                                          separator=',')
        
        # Calculates the accuracy of the language model on the dev set
        print("Results on dev set:")

        list_test_data = compiled_data.partition_by(
            'author'
        )

        table = []
        for i in list_test_data:
            author_name = i.item(0, 'author')

            total_count = i.shape[0]
            correct_count = i.filter(
                pl.col('prediction') == author_name
            ).shape[0]

            table.append([author_name, (correct_count/total_count) * 100])

            print(f"{author_name}\t\t{(correct_count/total_count) * 100:.1f}% correct")

        headers = ['Author', 'Accuracy']
        with open('./outputs/devset_result_discriminative.txt', 'w') as f:
            f.write(tabulate(table, headers, floatfmt='.1f'))

    else:
        # Print out in the format of what the author is line by line
        compiled_data = pl.DataFrame(
            {
                "text":test_text,
                "prediction":predicted_label
            }
        ).with_columns(
            pl.col('prediction').replace(id2label)
        )

        compiled_data.write_csv('./outputs/prediction_discriminative.csv',
                                separator=',')
        
        predicted_label = compiled_data['prediction'].to_list()

        for i in predicted_label:
            print(i)

    logging.info(f'Testing ended - Run Time: {time.time() - start_time}')

def run_discriminative(pl_text_data, bool_testing:bool):
    """
    Runs the discriminative author attribution model which is based on Roberta
    reference: 

    : param: pl_text_data = polars dataframe of the text data
    : param: bool_testing = boolean value indicating whether a testing file was provided (True) or not (False)
    """
    logging.info(f'Discriminative model run started')

    tokenized_author_data, id2label, data_collator = data_preprocessing(pl_text_data, bool_testing)
    trainer = train_discriminative(tokenized_author_data, id2label, data_collator)
    test_discriminative(tokenized_author_data, trainer, id2label, bool_testing)

    logging.info(f'Discriminative model run ended')