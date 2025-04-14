import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from pynvml import *
import os

import math

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def load_model(base_model_name:str,
               vocab_size:int=50_000,
               max_position_embddings=514,
               num_attention_heads=12,
               num_hidden_layers=12,
               type_vocab_size=1,):
    
    config = AutoConfig.from_pretrained(
        base_model_name,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )

    # kwargs={'vocab_size':vocab_size,
    #             'max_position_embeddings':max_position_embddings,
    #             'num_attention_heads':num_attention_heads,
    #             'num_hidden_layers':num_hidden_layers,
    #             'type_vocab_size':type_vocab_size}

    if torch.cuda.is_available():
        model = AutoModelForMaskedLM.from_config(config=config).cuda()
    else:
        model = AutoModelForMaskedLM.from_config(config=config)

    return model

def run_train(base_model_name:str,
             trained_tokenizer:str,
             trained_model:str,
             dataset,
             per_device_train_batch_size:int=64,
             per_device_eval_batch_size:int=32,
             gradient_accumulation_steps:int=2,) -> None:
    print_gpu_utilization()
    model = load_model(base_model_name=base_model_name)
    print_gpu_utilization()
    # tokenizer = ByteLevelBPETokenizer(vocab=trained_vocab, merges=trained_merge)
    # tokenizer._tokenizer.post_processor = BertProcessing(
    #     ("</s>", tokenizer.token_to_id("</s>")),
    #     ("<s>", tokenizer.token_to_id("<s>")),
    # )
    # tokenizer.enable_truncation(max_length=512)
    # tokenizer.save("./tokenizer.json")
    try:
        os.mkdir(trained_model)
    except:
        pass
    tokenizer = AutoTokenizer.from_pretrained(trained_tokenizer, max_length=512)

    tokenized_data = dataset['train'].map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=512), batched=True)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    max_steps = 500_000
    epoch_cnt = math.ceil(max_steps/len(tokenized_data))


    training_args = TrainingArguments(
        output_dir=trained_model,
        overwrite_output_dir=True,
        num_train_epochs=epoch_cnt,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        # gradient_accumulation_steps=gradient_accumulation_steps,
        # eval_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=True, 
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=6e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        warmup_steps=24_000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_data,
        preprocess_logits_for_metrics=(lambda logits, labels: torch.argmax(logits, dim=-1))
    )
    trainer.train()
    print_gpu_utilization()

    trainer.save_model(trained_model)

def test_trained(trained_tokenizer:str,
                 trained_model:str,
                 sample_sentence:str) -> None:
    fill_mask = pipeline(
        'fill-mask',
        model=trained_model,
        tokenizer=trained_tokenizer
    )

    print(fill_mask(sample_sentence))