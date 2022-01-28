# from transformers import pipeline

# summarisation = pipeline("summarization", model="facebook/bart-base", device="cuda")

# example = "Positive: Giovani great to meet you last week, Next step of interview process: Programming Challenge, Let us know when you are available"
# print(summarisation(example))
# print("dye")
from transformers.modeling_gpt2 import GPT2LMHeadModel
from pplm.run_pplm import run_pplm_example

model = "gpt2-medium"
_ = GPT2LMHeadModel.from_pretrained(model)
run_pplm_example(
    cond_text="Next step of interview process: Programming Challenge",
    num_samples=3,
    discrim="sentiment",
    class_label="very_positive",
    length=50,
    stepsize=0.03,
    sample=True,
    temperature=0.7,
    num_iterations=1,
    gamma=1,
    gm_scale=0.9,
    kl_scale=0.02,
    colorama=True,
    verbosity="quiet",
)

print("hm")
