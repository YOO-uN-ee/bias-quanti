from transformers import RobertaConfig, AutoConfig
config = RobertaConfig(
 vocab_size=50_000,
 max_position_embeddings=514,
 num_attention_heads=12,
 num_hidden_layers=12,
 type_vocab_size=1,
)

from transformers import RobertaTokenizer, AutoTokenizer
tokenizer = RobertaTokenizer.from_pretrained('/FashionHebBERT', max_length=512)

from transformers import RobertaForMaskedLM, AutoModelForMaskedLM
model = RobertaForMaskedLM(config=config).cuda()

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
 tokenizer=tokenizer,
 file_path='fashion.txt',
 block_size=128,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
 tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
 output_dir='/FashionBERT',
 overwrite_output_dir=True,
 num_train_epochs=3,
 per_device_train_batch_size=64,
 per_device_train_batch_size=128,
 save_steps=10_000,
 save_total_limit=2,
 learning_rate=6e-4,
 weight_decay=0.01,
 adam_beta1=0.9,
 adam_beta2=0.98,
 adam_epsilon=1e−6,
 warmup_steps=24_000,
)
trainer = Trainer(
 model=model,
 args=training_args,
 data_collator=data_collator,
 train_dataset=dataset,
)
trainer.train()

trainer.save_model('/FashionBERT')

from transformers import pipeline
fill_mask = pipeline(
 'fill-mask',
 model='/FashionBERT',
 tokenizer='/FashionBERT'
)
fill_mask('שמלת<mask>')