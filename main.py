import re
import gc
import demjson
import logging
import os
import copy
from flask import Flask, request
from flask_cors import *
import json
from model_deploy_interface import get_model_predict


def preprocess(data_slice):
    for i in range(len(data_slice)):
        data_slice[i] = [data_slice[i][0], data_slice[i][2], data_slice[i][3], data_slice[i][1]]
    return data_slice

def model_core(sentence, annotations):
    pass

def model_handler(kwargs):
    config = None
    with open('./config.json', 'r') as f:
        config = json.loads(f.read())
        config['lbs'] = kwargs['data']['labels']
        config['src_to_keep'] = kwargs['data']['sources']
    if config is not None:
        with open('./config.json', 'w') as f:
            json.dump(config, f)
    model_prediction = get_model_predict(demjson.decode(kwargs['data'].__str__()))
    gc.collect()
    datas = kwargs['data']['processedData']
    res = dict()
    res['text'] = ""
    res['source_files'] = ['ann', 'txt']
    res['annotation'] = []
    res['entities'] = []
    idx = 1
    for data, prediction in zip(datas, model_prediction):
        sentence = data['sentence']
        annotations = data['annotation']
        annotations = preprocess(annotations)
        res['annotation'].append(prediction)
        for key, value in prediction.items():
            key = key.replace('(', '')
            key = key.replace(')', '')
            key = key.replace(',', '')
            res['entities'].append(["T{}".format(idx), value[0], [[int(key.split(' ')[0]) + len(res['text']), int(key.split(' ')[1]) + len(res['text'])]]])
            idx += 1
        res['text'] += sentence + '\n'
    
    return res

app = Flask(__name__)

CORS(app, supports_credentials=True)

@app.route('/api/model/test', methods=['POST'])
def test():
    gc.collect()
    if request.method == 'POST':
        data = request.get_data(as_text=False)
        data_dict = json.loads(data)
        # print(data_dict)
        response = dict()
        response['status'] = 200
        try:
            response['data'] = model_handler(data_dict)
            gc.collect()
        except Exception as e:
            logging.exception('Exception occured while handling data')
            response['status'] = 503
        return response

@app.route('/api/model/train', methods=['POST'])
def train():
    if request.method == 'POST':
        data = request.get_data(as_text=False)
        data_dict = json.loads(data)['data']
        print(data_dict)
        os.system(data_dict['shell'])
        response = dict()
        response['status'] = 200
        try:
            response['data'] = model_handler(data_dict)
        except Exception as e:
            response['status'] = 503
        return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000")