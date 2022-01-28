downloads parser: git@github.com:webis-de/acl20-crawling-mailing-lists.git
use python 3.7.9
sudo apt-get install python3.7-dev
install requirements

# PARSE ENRON

**Use of original enron dataset**
download forked e-mail segmentation model:<https://github.com/Giovani-Merlin/acl20-crawling-mailing-lists>
first parse it keeping relevant information using <https://github.com/ZhaiResearchGroup/enron-parser>

use customized parser as we need the \n to keep the structure

* call email-parse with the enron dataset base path

```bash
python email_parser/parsing/message_segmenter.py train -f models/segmenter_hinge.epoch-11.val_loss-0.093.h5 -v annotations/enron-annotated-finetuning-validation.jsonl models/fasttext-model.bin annotations/enron-annotated-finetuning-train.jsonl ./annotations/enron-annoted.jsonl
```

parse - original kaggle dataset because parsed breaks the model...

```bash
python email_parser/parsing/message_segmenter.py predict models/segmenter_hinge.epoch-11.val_loss-0.093.h5 models/fasttext-model.bin data/enron_kaggle/emails.jsonl -o ./annotaded_enron.jsonl
```

Was needed to fix prediction, wtf results, saves as "text" of other examples
