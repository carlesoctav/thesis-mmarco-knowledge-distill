from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
parent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
parent_model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")

def embedding(datasets, parent_model, parent_tokenizer):

    def cls_pooling(model_output):
        return model_output.last_hidden_state[:,0]

    parent_model.to(device)
    

    def embedding_batch(examples):
        encoded_input = parent_tokenizer(examples["text_en"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            model_output = parent_model(**encoded_input)

        target_embedding = cls_pooling(model_output).detach().cpu().numpy()

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
            "input_ids_id": output_id.input_ids,
            "attention_mask_id": output_id.attention_mask,
        }

    tokenized_datasets = datasets.map(tokenize_batch, batched=True, num_proc=8)
    return tokenized_datasets




dataset = load_dataset("carles-undergrad-thesis/en-id-parallel-sentences")

embedding_dataset = embedding(dataset, parent_model, parent_tokenizer)
embedding_tokenized_dataset = tokenize(embedding_dataset, student_tokenizer)
embedding_tokenized_dataset.push_to_hub("carles-undergrad-thesis/en-id-parallel-sentences-embedding")
