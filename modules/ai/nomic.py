import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(text):
    # Replace "?" and "!" with "." to standardize sentence delimiters, ensuring not to remove necessary spaces.
    standardized_text = text.replace("?", ". ").replace("!", ". ")
    # Split the standardized text into sentences
    sentences = [sentence.strip() for sentence in standardized_text.split('.') if sentence]
    # Prepend "search_query: " to each sentence
    prefixed_sentences = ['search_query: ' + sentence for sentence in sentences]

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)
    model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True, rotary_scaling_factor=2)
    
    model.eval()

    # Tokenize sentences
    encoded_input = tokenizer(prefixed_sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        # Get model output
        model_output = model(**encoded_input)

    # Apply mean pooling to get sentence embeddings
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()