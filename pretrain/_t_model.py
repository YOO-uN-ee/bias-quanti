import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import pipeline

def load_model(base_model_name:str,
               vocab_size:int=50_000,
               max_position_embddings=514,
               num_attention_heads=12,
               num_hidden_layers=12,
               type_vocab_size=1,):
    
    config = AutoConfig(
        base_model_name,
        vocab_size= vocab_size,
        max_position_embeddings=max_position_embddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )

    if torch.cuda.is_available():
        model = AutoModelForMaskedLM(base_model_name, config=config).cuda()
    else:
        model = AutoModelForMaskedLM(base_model_name, config=config)

    return model

def pretrain(base_model_name:str,
             trained_tokenizer:str,
             trained_model:str,
             dataset,) -> None:

    model = load_model(base_model_name=base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(trained_tokenizer, max_length=512)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=trained_model,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
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
        train_dataset=dataset,
    )
    trainer.train()

    trainer.save_model(trained_model)

def test_trained(trained_tokenizer:str,
                 trained_model:str,
                 sample_sentence:str) -> None:
    fill_mask = pipeline(
        'fill-mask',
        model=model_dir,
        tokenizer=tokenizer_dir
    )

    print(fill_mask(sample_sentence))