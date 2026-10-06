# FastAPI - Python based web framework

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


T5model = T5ForConditionalGeneration.from_pretrained("./Saved_Model")
tokenizer = T5Tokenizer.from_pretrained("./Saved_Model")

T5model.to(device)
T5model.eval()

template = Jinja2Templates(directory=".")


class Dialogue(BaseModel):
    dialogue: str



def clean_data(text):

    if pd.isna(text):
        return ""

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text

def summarize_dialogue(dialogue):

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        target = T5model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        target[0],
        skip_special_tokens=True
    )

    return summary


@app.post("/summarize/")
async def summarize(dialogue: Dialogue):

    summary = summarize_dialogue(dialogue.dialogue)

    return {
        "summary": summary
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return template.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )