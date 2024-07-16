import os
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# Disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Check CPU availability
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

# Load the BERT model and tokenizer
model_name = 'bert-base-uncased'
model = TFBertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load and preprocess the dataset
dataset = load_dataset('glue', 'mrpc')

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the dataset for TensorFlow
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('tensorflow')

train_dataset = tokenized_datasets['train'].to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'labels'],
    shuffle=True,
    batch_size=8,
    collate_fn=lambda x: {'input_ids': tf.ragged.constant([f['input_ids'] for f in x]).to_tensor(),
                          'attention_mask': tf.ragged.constant([f['attention_mask'] for f in x]).to_tensor(),
                          'labels': tf.constant([f['labels'] for f in x])}
)

validation_dataset = tokenized_datasets['validation'].to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'labels'],
    shuffle=False,
    batch_size=8,
    collate_fn=lambda x: {'input_ids': tf.ragged.constant([f['input_ids'] for f in x]).to_tensor(),
                          'attention_mask': tf.ragged.constant([f['attention_mask'] for f in x]).to_tensor(),
                          'labels': tf.constant([f['labels'] for f in x])}
)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
model.fit(train_dataset, validation_data=validation_dataset, epochs=3)

# Evaluate the model
model.evaluate(validation_dataset)

# Save the model
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
