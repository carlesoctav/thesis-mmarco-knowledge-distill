import tensorflow as tf
from transformers import TFXLMRobertaModel, AutoTokenizer, TFAutoModel, TFDistilBertModel
from datasets import load_dataset
from datetime import datetime
import logging
from pyprojroot.here import here
import os 


model = TFAutoModel.from_pretrained("distilbert-base-multilingual-cased",from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

weights = model.get_weights()


model.load_weights("disk/model/last_epoch/2023-10-20_02-16-33.h5")


model.push_to_hub(
    "carles-undergrad-thesis/distillbert-tasb-en-id-mmarco-knowledge-distillation",
)

tokenizer.push_to_hub(
    "carles-undergrad-thesis/distillbert-tasb-en-id-mmarco-knowledge-distillation",
)

