#!/bin/bash

uvicorn \
--host 0.0.0.0 \
--port 6060 \
--workers 1 \
--log-level debug \
email_writer.main:app