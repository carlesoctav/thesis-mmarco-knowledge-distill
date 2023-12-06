from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
import torch_xla.core.xla_model as xm
device = xm.xla_device()

student_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased", use_fast=True)
parent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5", use_fast=True)
parent_model = AutoModel.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")

def embedding(datasets, parent_model, parent_tokenizer):

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    parent_model.to(device)
    
    def embedding_batch(examples):
        encoded_input = parent_tokenizer(examples["text_en"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            model_output = parent_model(**encoded_input)

        target_embedding = mean_pooling(model_output, encoded_input["attention_mask"]).detach().cpu().numpy()

        return {
            "target_embedding": target_embedding
        }

    embedding_datasets = datasets.map(embedding_batch, batched=True,batch_size=384)
    return embedding_datasets

        
def tokenize(datasets, student_tokenizer):
    """
    datasets: huggingface datasets
    student_tokenizer: huggingface tokenizer (student tokenizer)
    """
    def tokenize_batch(examples):
        """
        batch tokenize function
        """
        output_en = student_tokenizer(examples["text_en"], padding="max_length", truncation=True, max_length=256)
        output_id = student_tokenizer(examples["text_id"], padding="max_length", truncation=True, max_length=256)

        return {
            "input_ids_en": output_en.input_ids,
            "attention_mask_en": output_en.attention_mask,
            "token_type_ids_en": output_en.token_type_ids, 
            "input_ids_id": output_id.input_ids,
            "attention_mask_id": output_id.attention_mask,
            "token_type_ids_id": output_id.token_type_ids,
        }

    tokenized_datasets = datasets.map(tokenize_batch, batched=True, num_proc=90)
    return tokenized_datasets




dataset = load_dataset("carles-undergrad-thesis/en-id-parallel-sentences")

embedding_dataset = embedding(dataset, parent_model, parent_tokenizer)
embedding_tokenized_dataset = tokenize(embedding_dataset, student_tokenizer)
embedding_tokenized_dataset.push_to_hub("carles-undergrad-thesis/en-id-parallel-sentences-embedding-vbert")
