# E-mail writer

## Usage

Create models directory on your project root then download [this](https://drive.google.com/drive/folders/1avMygQ9JnPTduFTvLriIxbVah-G6yqm0?usp=sharing) trained model on AESLC dataset and put it in the models directory.
Then, you can run it as a container:

* Build the image `docker build -f email_writer.Dockerfile -t email_writer .` and run it with `docker run -p 6060:6060 --name email_writer email_writer`. To enable gpu, install [container-toolki](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and run the container with `docker run --gpus all -p 6060:6060 --name email_writer email_writer`.
* Or you can run it direct by setting up the repo (preferably inside a venv) with `pip install -r requirements.txt` and `pip install -e email_writer`
  * Then, run the command `uvicorn email_writer.main:app --port 6060 --host 0.0.0.0`

To call the inference endpoint, just call `http://0.0.0.0:6060/email` with a GET request and header of `Content-Type=application/json` with the needed json body or directly by using swagger at `http://0.0.0.0:6060/docs#/default/read_root_email_get`.

## Endpoint docs

* subject: subject of the email
* from: sender of the email
* to: receiver of the email
* salutation: salutation of the sender
* temperature: temperature used by model inference
* n_gen: number of examples generated

Example request:

```Json
{
"subject": "Interview Challenge",
"salutation": "Giovani nice to meet you last week",
"from": "employer@host.com",
"to": "candidate@host.com",
"temperature": 0.7,
"n_gen": 4,
}
```
