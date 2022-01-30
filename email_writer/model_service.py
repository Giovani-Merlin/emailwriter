import logging
import torch

from models import GPT2Model

model_class = GPT2Model()
tokenizer, model = model_class.gpt2_enron

logger = logging.getLogger(__name__)


def custom_tokenizer(subject, text):
    added = subject + tokenizer.sep_token
    added_inputs = tokenizer(added, return_tensors="pt", padding=True)
    if text != "":
        text_inputs = tokenizer(text, return_tensors="pt", padding=True)
        output = {}
        output["input_ids"] = torch.hstack([added_inputs["input_ids"], text_inputs["input_ids"]]).to(model.device)
        output["attention_mask"] = torch.hstack([added_inputs["attention_mask"], text_inputs["attention_mask"]]).to(
            model.device
        )
    else:
        output = added_inputs.to(model.device)
        text_inputs = ""
    return output, len(added_inputs), len(text_inputs)


def generate_body(subject, text="", temperature=0.7, n_gen=1):
    inputs, added_size, text_size = custom_tokenizer(subject, text)

    model_outputs_init = model.generate(
        **inputs,
        random_seed=0,
        do_sample=True,
        max_length=text_size + 80,
        min_length=text_size + 20,
        top_k=15,
        temperature=temperature,
        num_return_sequences=n_gen,
    )
    output = {}
    for i, model_output in enumerate(model_outputs_init):
        output[i] = "".join([tokenizer.decode(t, skip_special_tokens=True) for t in model_output]).lstrip(subject)
    return output


if __name__ == "__main__":
    subject = "Next step of interview process: Programming Challenge"
    text = "Giovani great to meet you last week"
    body = generate_body(subject, text)
    for output in body.values():
        print("\n")
        print(output)
