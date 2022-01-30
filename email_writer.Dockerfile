FROM python:3.7

ENV MODEL_PATH=models
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
#COPY set_repo.sh . 
#RUN sh set_repo.sh
# ----
# -----

#RUN mkdir models
COPY models models
COPY email_writer email_writer
RUN pip install -e email_writer

EXPOSE 6060

# Entrypoint
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

#ENTRYPOINT ["/bin/bash ls"]
ENTRYPOINT ["/app/docker-entrypoint.sh"]