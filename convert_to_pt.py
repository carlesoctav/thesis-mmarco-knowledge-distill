from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("carles-undergrad-thesis/distillbert-tasb-en-id-mmarco-knowledge-distillation", from_tf=True,)

model.push_to_hub(
    "carles-undergrad-thesis/distillbert-tasb-en-id-mmarco-knowledge-distillation",
)

