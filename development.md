# Development

This markdown file aims to keep track of each step in the creation of such a repository, from the research stage of the problem to its implementation in a cloud system.

## Research

**3 hours**:  Basic understanding of the task, search about generative models, base model's pappers and data sets.

**5 hours**: Deeper exploration

* [GPT-2](http://jalammar.github.io/illustrated-gpt2/) (355M medium, 127M small, 87M small distilled)
* [Decoding Methods](https://huggingface.co/blog/how-to-generate)
* [PPLM](https://eng.uber.com/pplm/) - control technique for gpt2
* [CTRL](https://github.com/salesforce/ctrl) (1.6B)
* [T5](https://towardsdatascience.com/data-to-text-generation-with-t5-building-a-simple-yet-advanced-nlg-model-b5cce5a6df45) (60M small, 220 Base)
* [Newer control technique](https://aclanthology.org/2021.findings-emnlp.194.pdf)
* [Simple gpt-2 with discriminator](https://bonkerfield.org/2020/02/combining-gpt-2-and-bert/s)
* [Updated review in control techniques](https://lilianweng.github.io/lil-log/2021/01/02/controllable-neural-text-generation.html)

### Planning

**? hours, 2 maybe**: "Pre-train" model on emails dataset to get the LGM to follow the email structure, then customize output as:

1. Basic e-mail informations: from, to, subject.
2. Subjectc fields for creating the text
    1. Or a list of subjects to be used as a template of each paragraph (or whole e-mail)
    2. Or a pre-defined list of field to use as template, as:
        * Context (or "init"), main subject, objective.
3. Tone or topic (positive, negative, neutral, comercial, etc)

By 1 we could make specific modes for each person or associate a tone to each person (by using some controlling model or custom tokens. Here we will just use to create the email structure using a template.

In 2 we will try both implementations. 1.1 implementation needs custom tokens/control for each paragraph (thus, harder), 1.2 implementation needs just to use the field as prompt of each paragraph (thus, easier).
T5 model is more flexible and do all the e-mail directly, GPT-2 needs to divide the e-mail into paragraphs to take in account all the needed fields.

In 3 we will use an objective field to define the tone of the e-mail (in practice, just one more controlling field)

## Data

Needs to generate text in e-mail like format.

**2 hours**:

* [Personal mail corpus - stackexchange](https://opendata.stackexchange.com/questions/4517/obtaining-personal-mail-corpus)
* [Webcrawling of open archive email lists](https://github.com/webis-de/acl20-crawling-mailing-lists) - would be the best one, but needs to request access.*

  * In fact, with further checking, it can be found in internet archive <https://archive.org/details/webis-gmane-19?&sort=-week&page=2> but would need to filter it for the desired archives...
* [Git Hub](https://github.com/Mithileysh/Email-Datasets)

### Enron

Choice of [Enron email dataset](https://www.cs.cmu.edu/~./enron/) for pre-training as it has a significant size ( > 500k emails and 1.4GB) and is more business oriented and has some [libraries](https://github.com/ZhaiResearchGroup/enron-parser) to parse it.

#### Dataset preparation

### Training

[gpt2, gpt-neo, t5 for text classification](https://pasaentuciudad.com.mx/guide-to-fine-tuning-text-generation-models-gpt-2-gpt-neo-and-t5/)
[optimize gpt-2 and t5 for nvidia](https://developer.nvidia.com/blog/optimizing-t5-and-gpt-2-for-real-time-inference-with-tensorrt/)

### For later


