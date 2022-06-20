FROM ufoym/deepo:cpu
RUN pip install flask torch transformers flask_cors demjson pytokenizations && echo "import nltk\nnltk.download('punkt')" >> prehandle.py && python prehandle.py && rm prehandle.py
