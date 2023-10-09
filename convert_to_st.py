from sentence_transformers import SentenceTransformer, models


max_seq_length = 128  # Student model max. lengths for inputs (number of word pieces)
student_model_name = "carles-undergrad-thesis/distillbert-en-id-mmarco-knowledge-distillation"

word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

student_model.save_to_hub(
    repo_name="st-distillbert-en-id-mmarco-knowledge-distillation",
    organization="carles-undergrad-thesis",
)
