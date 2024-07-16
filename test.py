import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# Load the BERT model and tokenizer
model_name = 'bert-base-uncased'
model = TFBertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Example text
text = ["Hello, how are you?"]

# Tokenize the text
inputs = tokenizer(text, return_tensors='tf')

# Perform a forward pass
outputs = model(**inputs)

# Print the outputs
print(outputs)
