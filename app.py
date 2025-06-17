# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Load model & tokenizer
model_path = "./ner_model"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

# Load tag2id and id2tag
unique_tags = sorted(tokenizer.get_vocab().keys())  # You can also save/load tag2id
# For now, assume same tag2id/id2tag as training
tag2id = {tag: idx for idx, tag in enumerate(model.config.id2label.values())}
id2tag = {v: k for k, v in tag2id.items()}

# FastAPI App
app = FastAPI()

# Request model
class NERRequest(BaseModel):
    sentence: str

# Inference Endpoint
@app.post("/predict")
async def predict(req: NERRequest):
    sentence = req.sentence.strip().split()  # split into tokens

    # Tokenize input
    inputs = tokenizer([sentence], is_split_into_words=True, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Map predictions to tags
    pred_tags = []
    for idx, word_id in enumerate(inputs.word_ids(batch_index=0)):
        if word_id is not None:
            tag = model.config.id2label[predictions[0][idx].item()]
            pred_tags.append({"word": sentence[word_id], "tag": tag})

    return {"entities": pred_tags}
