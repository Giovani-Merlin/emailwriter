from enum import Enum
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model_service import generate_body

app = FastAPI()


class Tone(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class inference_api_request(BaseModel):
    # Model fields
    subject: str
    # tone: Tone
    salutation: str  # Optional[str] = Field(None, title="Salutation to use")
    # closing: Optional[str] = Field(None, title="Closing to use")
    from_: str = Field(None, title="From address to use", alias="from")
    to: str  # Optional[str] = Field(None, title="To address to use")
    temperature: Optional[float] = 0.7
    n_gen: Optional[int] = 3

    class Config:
        schema_extra = {
            "example": {
                "subject": "Interview Challenge",
                "salutation": "Giovani nice to meet you last week",
                "from": "employer@host.com",
                "to": "candidate@host.com",
                "temperature": 0.7,
                "n_gen": 4,
            }
        }


def parse_email(from_, to, subject, body):
    email = f"from: {from_}\n"
    email += f"to: {to}\n"
    email += f"subject: {subject}\n\n"
    email += body
    return email


@app.get("/")
def preview():
    return {"usage": "This app hands out the semi-automatic e-mail creation."}


@app.get("/email")
def read_root(body: inference_api_request):
    subject = body.subject
    salutation = body.salutation
    temperature = body.temperature
    n_gen = body.n_gen
    body_texts = generate_body(subject, salutation, temperature, n_gen)
    from_ = body.from_
    to = body.to
    for key, output in body_texts.items():
        body_texts[key] = parse_email(from_, to, subject, output)
    return body_texts
