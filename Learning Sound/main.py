import os
from utils import model_url, data_collator, compute_metrics, load_data, processor
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from huggingface_hub import login
import wandb

wandb.login(key=os.environ["WANDB_API_KEY"])
login(token=os.environ["HUGGINGFACE_API_KEY"])
model = WhisperForConditionalGeneration.from_pretrained(model_url)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
batch_size = 8
collator = data_collator
train_data, valid_data = load_data()

training_args = Seq2SeqTrainingArguments(
    output_dir="./yoruba_whisper",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    generation_max_length=255,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard", "wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.push_to_hub("training complete")
