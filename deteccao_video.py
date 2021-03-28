from __future__ import division

import argparse
import json
from json.decoder import JSONDecodeError
from operator import itemgetter

import cv2

from models import *
from utils.datasets import *
from utils.utils import *


def converter_rgb(img):
    # Converter Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def converter_bgr(img):
    # Converter red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def build_response(message_id, measurements):
    """
    :param message_id: The message identifier who requested the test
    :param measurements: the measurements used in the validation
    :return: dict
    """
    protocol = {
        "id": message_id,
        "response": measurements
    }
    return protocol


class Detect:
    def __init__(self, options):
        self.opt = options
        if opt.debug:
            print(opt)
        self._start_device()
        self._start_model()
        self._start_output()

    def receive_protocol(self, message):
        try:
            data = json.loads(message)
            identifier = data["id"].split("@")[0]
            return identifier, data
        except JSONDecodeError:
            return False, {"action": "unknown"}
        except KeyError:
            return False, {"action": "protocol needs an identifier"}

    def _start_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.opt.debug:
            print("cuda" if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def _start_model(self):
        self.model = Darknet(opt.model_def, img_size=opt.img_size).to(self.device)
        if self.opt.weights_path.endswith(".weights"):
            self.model.load_darknet_weights(opt.weights_path)
        else:
            self.model.load_state_dict(torch.load(opt.weights_path))
        self.model.eval()
        self.classes = load_classes(opt.class_path)

    def _start_output(self):
        if self.opt.webcam == 1:
            self.cap = cv2.VideoCapture(0)
            # cap.set(10, 180)
            self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                       (1920, 1080))  # (1280, 960)
        else:
            self.cap = cv2.VideoCapture(self.opt.directorio_video)
            # frame_width = int(cap.get(3))
            # frame_height = int(cap.get(4))
            self.out = cv2.VideoWriter('outp.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1920, 1080))
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")

    def stop(self):
        self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self, identifier):
        while self.cap:
            ret, frame = self.cap.read()
            if ret is False:
                return
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            # A imagem vem em BGR e é convertida em RGB que é o que o modelo requer
            RGBimg = converter_rgb(frame)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(imgTensor)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            lista_mola = []
            lista_pino = []

            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        if self.classes[int(cls_pred)] == 'mola':
                            classes_dict = {'dist': int(x1), 'classe': self.classes[int(cls_pred)]}
                            lista_mola.append(classes_dict)

                        if self.classes[int(cls_pred)] == 'pino':
                            classes_dict = {'dist': int(x1), 'classe': self.classes[int(cls_pred)]}
                            lista_pino.append(classes_dict)

                    # lista ordenada para mola
                    sorted_lista_mola = sorted(lista_mola, key=itemgetter('dist'))
                    # json_convert_mola = json.dumps(sorted_lista_mola, indent=4)
                    # print(json_convert_mola)

                    # lista ordenada para pino

                    sorted_lista_pino = sorted(lista_pino, key=itemgetter('dist'))
                    # json_convert_pino = json.dumps(sorted_lista_pino, indent=4)
                    # print(json_convert_pino)
                    # conversao dict para json format

                    resultado = {
                        "molas": self.classify_molas(sorted_lista_mola),
                        "pinos": self.classify_pinos(sorted_lista_pino),
                    }

                    return build_response(identifier, resultado)

    def classify_molas(self, molas):
        if len(molas) == 2:
            # print('status mola ---- OK')
            dict_mola = {'tipo': 'mola', 'status': 'APROVADO', 'QTD': len(molas)}
            # print(json.dumps(dict_mola, indent=4))
        else:
            dict_mola = {'tipo': 'mola', 'status': 'NG', 'QTD': len(molas)}
            # print(json.dumps(dict_mola, indent=4))
        return dict_mola

    def classify_pinos(self, pinos):
        if len(pinos) == 4:
            # print('status pino ---- OK')
            dict_pino = {'tipo': 'pino', 'status': 'APROVADO', 'QTD': len(pinos)}
            # print(json.dumps(dict_pino, indent=4))
        else:
            dict_pino = {'tipo': 'pino', 'status': 'NG', 'QTD': len(pinos)}
            # print(json.dumps(dict_pino, indent=4))
        return dict_pino


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=0, help="Is the video processed video? 1 = Yes, 0 == no")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, default="videos/fhd_c_ilum.mp4", help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--show_result", type=bool, default=False, help="after the detection, should display the image")
    parser.add_argument("--debug", type=bool, default=False, help="print debug messages in the execution")
    opt = parser.parse_args()

    # Starts detection object
    detection = Detect(opt)

    # Waits for a command as protocol
    try:
        while True:
            # Receive command
            identifier, receive_cmd = detection.receive_protocol(input())

            if receive_cmd["action"] == "test":
                # Execute detection
                result = detection.run(identifier)

                # Send the result of detection to STDOUT
                print(json.dumps(result))
            elif receive_cmd["action"] == "ping":
                print(json.dumps({
                    "id": identifier,
                    "action": "ping",
                    "response": "pong"
                }))
            else:
                print(json.dumps({
                    "action": "unknown"
                }))

    except KeyboardInterrupt:
        # Received interruption
        detection.stop()
