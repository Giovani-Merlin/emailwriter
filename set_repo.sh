wget https://codeload.github.com/uber-research/PPLM/zip/refs/heads/master -O pplm.zip
unzip pplm.zip && mv PPLM-master pplm
touch pplm/__init__.py
#source ../venv/bin/activate && 

pip install -r requirements.txt