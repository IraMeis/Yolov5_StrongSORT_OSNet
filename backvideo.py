"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import glob
import os
import shutil
import uuid
from pathlib import Path
from flask import send_file, Flask, request, jsonify
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
curDir = 'runs/track'
evalDir = 'tracks'
temp = Path("temp")

DETECTION_URL = "/api/nets/run/<model>"
EVAL_URL = "/api/nets/eval/<uuid>"


@app.route(DETECTION_URL, methods=["POST"])
@cross_origin(expose_headers=['Content-Disposition'])
def predict(model):

    if request.files.get("video"):
        curName = str(uuid.uuid4())
        while os.path.exists(ROOT / curDir / curName):
            curName = str(uuid.uuid4())
        vi_file = request.files["video"]
        os.mkdir(ROOT / temp / curName)
        vf = ROOT / temp / curName / vi_file.filename
        vi_file.save(vf)

        if model in models:
            def internal_parser_args():
                in_parser = argparse.ArgumentParser(description="")
                in_parser.add_argument('--yolo-weights', nargs='+', type=Path, default=ROOT / Path('yolov5') / Path(model + '.pt'), help='model.pt path(s)')
                in_parser.add_argument('--strong-sort-weights', type=Path, default=ROOT / 'weights/osnet_x0_25_msmt17.pt')
                in_parser.add_argument('--source', type=str, default=vf, help='file/dir/URL/glob, 0 for webcam')
                in_parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
                in_parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
                in_parser.add_argument('--name', default=curName, help='save results to project/name')
                in_parser.add_argument('--project', default=ROOT / curDir, help='save results to project/name')
                # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                in_parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
                in_parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
                return in_parser.parse_args()

            track.main(internal_parser_args())
            dirExp = ROOT / curDir / curName
            delete_dir(ROOT / temp / curName)
            old_file = glob.glob(str(dirExp) + "/*.webm")[0]
            new_file = os.path.join(dirExp, curName + ".webm")
            os.rename(old_file, new_file)

            return send_file(new_file)

    return "BAD REQUEST", status.HTTP_400_BAD_REQUEST


@app.route(EVAL_URL, methods=["GET"])
@cross_origin()
def eval(uuid):
    mainPath = ROOT / curDir / uuid
    d = {}
    try:
        with open(glob.glob(str(mainPath / evalDir) + "/*.eval")[0]) as f:
            for line in f:
                (key, val) = line.split()
                d[int(key)] = val
        dam_count = {'D00': 0, 'D10': 0, 'D20': 0, 'D40': 0, 'ALL': 0}
        for k, v in d.items():
            try:
                dam_count[v] += 1
                dam_count['ALL'] += 1
            except KeyError:
                return "", status.HTTP_406_NOT_ACCEPTABLE
        return jsonify(dam_count)

    except IndexError:
        return "", status.HTTP_204_NO_CONTENT

    finally:
        delete_dir(mainPath)


def delete_dir(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (path, e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5001, type=int, help="port number")
    opt = parser.parse_args()

    app.run(host="0.0.0.0", port=opt.port)
