# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Load model & tokenizer
model_path = "./ner_model"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

# id2tag mapping — must match training
id2tag = {
    0: 'B-art',
    1: 'B-eve',
    2: 'B-geo',
    3: 'B-gpe',
    4: 'B-nat',
    5: 'B-org',
    6: 'B-per',
    7: 'B-tim',
    8: 'I-art',
    9: 'I-eve',
    10: 'I-geo',
    11: 'I-gpe',
    12: 'I-nat',
    13: 'I-org',
    14: 'I-per',
    15: 'I-tim',
    16: 'O'
}

# FastAPI app
app = FastAPI()

# Request model
class NERRequest(BaseModel):
    sentence: str

# Inference Endpoint
@app.post("/predict")
async def predict(req: NERRequest):
    sentence = req.sentence.strip()

    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []

    for idx, token in enumerate(tokens):
        if token.startswith("##") or token in ["[CLS]", "[SEP]"]:
            continue

        tag = id2tag[predictions[0][idx].item()]
        entities.append({"word": token, "tag": tag})

    return {"entities": entities}
