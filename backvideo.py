"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import datetime
import glob
import os
import shutil
import uuid
from logging.config import fileConfig
from pathlib import Path
from flask import send_file, Flask, request, jsonify
from flask_apscheduler import APScheduler
from flask_cors import CORS, cross_origin
from flask_api import status
import track

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

paths = []
models = ["model1", "model3", "model2", "model4"]

curDir = ROOT / Path('runs/track')
temp = ROOT / Path('temp')
sortPath = ROOT / Path('weights/osnet_x0_25_msmt17.pt')
yoloPath = ROOT / Path('yolov5')
evalDir = 'tracks'
pathsDates = []

DETECTION_URL = "/api/nets/run/<model>"
EVAL_URL = "/api/nets/eval/<uuid>"

scheduler = APScheduler()
scheduler.init_app(app)

fileConfig('logging.cfg')


@scheduler.task('interval', id='deleteVideos', minutes=5, misfire_grace_time=300)
def deleteVideos():
    for pd in pathsDates:
        try:
            if pd[0] == temp:
                timeSeconds = 600
            else:
                timeSeconds = 80
            if (datetime.datetime.now() - pd[2]).seconds > timeSeconds:
                shutil.rmtree(pd[1])
                pathsDates.remove(pd)
        except FileNotFoundError:
            pathsDates.remove(pd)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (pd, e))


@app.route(DETECTION_URL, methods=["POST"])
@cross_origin(expose_headers=['Content-Disposition'])
def predict(model):
    if request.files.get("video"):
        curName = str(uuid.uuid4())
        while os.path.exists(curDir / curName):
            curName = str(uuid.uuid4())
        vi_file = request.files["video"]
        os.makedirs(temp / curName)
        vf = temp / curName / vi_file.filename
        vi_file.save(vf)
        pathsDates.append((temp, temp / curName, datetime.datetime.now()))

        if model in models:
            track.run(yolo_weights=yoloPath / Path(model + '.pt'),
                      strong_sort_weights=sortPath,
                      source=vf,
                      name=curName,
                      project=curDir,
                      save_txt=True,
                      save_vid=True,
                      conf_thres=0.5,
                      iou_thres=0.5)
            dirExp = curDir / curName
            pathsDates.append((curDir, dirExp, datetime.datetime.now()))
            delete_dir(temp / curName)
            try:
                old_file = glob.glob(str(dirExp) + "/*.webm")[0]
                new_file = os.path.join(dirExp, curName + ".webm")
                os.rename(old_file, new_file)
                return send_file(new_file)
            except IndexError:
                return "NO CONTENT", status.HTTP_204_NO_CONTENT

    return "BAD REQUEST", status.HTTP_400_BAD_REQUEST


@app.route(EVAL_URL, methods=["GET"])
@cross_origin()
def eval(uuid):
    mainPath = curDir / uuid
    d = {}
    dam_count = {'D00': 0, 'D10': 0, 'D20': 0, 'D40': 0, 'ALL': 0}
    try:
        with open(glob.glob(str(mainPath / evalDir) + "/*.eval")[0]) as f:
            for line in f:
                (key, val) = line.split()
                d[int(key)] = val
        for k, v in d.items():
            try:
                dam_count[v] += 1
                dam_count['ALL'] += 1
            except KeyError as e:
                app.logger.error('Unexpected predictions. %s' % e)
                return "", status.HTTP_406_NOT_ACCEPTABLE
        return jsonify(dam_count)

    except IndexError:
        return jsonify(dam_count)

    finally:
        delete_dir(mainPath)


def delete_dir(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (path, e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5001, type=int, help="port number")
    opt = parser.parse_args()
    scheduler.start()
    app.run(host="0.0.0.0", port=opt.port)
else:
    scheduler.start()
