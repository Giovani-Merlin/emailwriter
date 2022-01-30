import logging
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class GPT2Model(object):
    @property
    def gpt2_enron(self):
        PATH = "models"  # os.environ["MODEL_PATH"]
        tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)

        logger.info("Loading model")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model device {self.device}")
        model = AutoModelForCausalLM.from_pretrained(PATH, local_files_only=True).to(self.device)

        logger.info("Model successfully loaded")

        return tokenizer, model
