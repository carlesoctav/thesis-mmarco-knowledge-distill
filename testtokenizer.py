from transformers import AutoTokenizer

student_tokenizer = AutoTokenizer.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")

output_en = student_tokenizer("hello", padding="max_length", truncation=True, max_length=256)

print(output_en)