from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch_xla.core.xla_model as xm
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

device = xm.xla_device()
print(f"==>> device: {device}")




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




