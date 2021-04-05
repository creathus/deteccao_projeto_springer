from __future__ import division

import argparse
import json
from collections import defaultdict
from json.decoder import JSONDecodeError
from operator import itemgetter
import itertools

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


def calculate_centr(coord):
    """
    Function to calculate the centroid
    :param coord:
    """
    return coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2)


def calculate_centr_distances(centroid_1, centroid_2):
    """
    Calculate distance between 2 centroids
    :param centroid_1:
    :param centroid_2:
    """
    return math.sqrt((centroid_2[0] - centroid_1[0]) ** 2 + (centroid_2[1] - centroid_1[1]) ** 2)


def calculate_perm(centroids):
    """
    Calculate all permutations between the centroids
    :param centroids:
    :return:
    """
    permutations = []
    for current_permutation in itertools.permutations(centroids, 2):
        if current_permutation[::-1] not in permutations:
            permutations.append(current_permutation)
    return permutations


def midpoint(p1, p2):
    """
    Calculate the middle point between 2 points
    :param p1:
    :param p2:
    """
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


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
    def __init__(self, options, width=1920, height=1080):
        self.width = width
        self.height = height
        self.dim = (self.width, self.height)
        self.opt = options
        if opt.debug:
            print(opt)
        self._start_device()
        self._start_model()
        self._start_input_and_output()

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

    def _start_input_and_output(self):
        if self.opt.webcam == 1:
            self.cap = cv2.VideoCapture(0)
            # cap.set(10, 180)
            self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                       self.dim)  # (1280, 960)
        else:
            self.cap = cv2.VideoCapture(self.opt.directorio_video)
            # frame_width = int(cap.get(3))
            # frame_height = int(cap.get(4))
            self.out = cv2.VideoWriter('outp.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, self.dim)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")

    def stop(self):
        self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self, identifier, only_first_detection=True):
        while self.cap:
            ret, frame = self.cap.read()
            if ret is False:
                return
            frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_CUBIC)
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
                    boxes = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                    molas_centroids = []
                    pinos_centroids = []
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls_conf, cls_pred = box
                        box_w = x2 - x1
                        box_h = y2 - y1
                        coord = [x1, y1, box_w, box_h]
                        # Calculate the center of box
                        centroid = calculate_centr(coord)

                        if self.opt.show_result:
                            color = [int(c) for c in self.colors[int(cls_pred)]]
                            frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)
                            cv2.putText(frame, self.classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        color,
                                        3)  # Nome da classe detectada
                            cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        color, 3)  # Certeza de predição da classe

                        if self.classes[int(cls_pred)] == 'mola':
                            molas_centroids.append(centroid)
                            classes_dict = {'dist': int(x1), 'classe': self.classes[int(cls_pred)]}
                            lista_mola.append(classes_dict)

                        if self.classes[int(cls_pred)] == 'pino':
                            pinos_centroids.append(centroid)
                            classes_dict = {'dist': int(x1), 'classe': self.classes[int(cls_pred)]}
                            lista_pino.append(classes_dict)

                    # lista ordenada para mola
                    sorted_lista_mola = sorted(lista_mola, key=itemgetter('dist'))
                    sorted_lista_pino = sorted(lista_pino, key=itemgetter('dist'))

                    resultado = {
                        "molas": self.classify_molas(sorted_lista_mola, molas_centroids),
                        "pinos": self.classify_pinos(sorted_lista_pino, pinos_centroids),
                    }
                    if self.opt.show_result:
                        print(json.dumps(resultado, indent=4))

                    if only_first_detection:
                        return build_response(identifier, resultado)

            if self.opt.show_result:
                # Convertemos de volta a BGR para que OpenCV possa colocar nas cores corretas
                if opt.webcam == 1:
                    cv2.imshow('frame', converter_bgr(RGBimg))
                    self.out.write(RGBimg)
                else:
                    self.out.write(converter_bgr(RGBimg))
                    cv2.imshow('Salcomp Processo', RGBimg)
                    # cv2.waitKey(0)
                    # Pressione Q no teclado para terminar o processo de execução do algoritmo (quit)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        if self.opt.show_result:
            self.out.release()
            self.cap.release()
            cv2.destroyAllWindows()

    def classify_molas(self, molas, centroids):
        qty = len(molas)
        status = "APROVADO" if qty == 2 else "NG"
        return {'tipo': 'mola', 'status': status, 'QTD': qty}

    def classify_pinos(self, pinos, centroids):
        qty = len(pinos)
        raw_data = defaultdict(dict)
        # Ordenar centros pelo eixo x, primeiro par é esquerdo e o segundo par é direito.
        sorted_pinos = sorted(centroids, key=lambda x: x[0])
        for i, centroid_pairs in enumerate(zip(*[iter(sorted_pinos)] * 2)):
            pino_pairs = calculate_perm([*centroid_pairs])
            for pairs in pino_pairs:
                dist = calculate_centr_distances(pairs[0], pairs[1])
                # primeiro par é esquerdo e o segundo par é direito.
                which = "left" if i == 0 else "right"
                raw_data[which]["pin_a"] = {"x": float(pairs[0][0]), "y": float(pairs[0][1])}
                raw_data[which]["pin_b"] = {"x": float(pairs[1][0]), "y": float(pairs[1][1])}
                raw_data[which]["diff"] = float(dist)

        # Existe a quantidade correta de PINOS ?
        is_quantity_ok = qty == 4
        # A distâncias entre os pinos da esquerda está dentro do aceitável ?
        is_left_ok = self.verify_pinos_distance(raw_data["left"]["diff"])
        # A distâncias entre os pinos da direita está dentro do aceitável ?
        is_right_ok = self.verify_pinos_distance(raw_data["right"]["diff"])
        # Se tudo estiver correto, então APROVADO, se não REPROVADO
        status = "APROVADO" if is_quantity_ok and is_left_ok and is_right_ok else "NG"
        return {'tipo': 'pino', 'status': status, 'QTD': qty, "raw": raw_data}

    def verify_pinos_distance(self, distance):
        # TODO: Inferir a distância entre os pinos e classificar como PASS OR FAIL
        # TODO: Precisar estar entre o MÁXIMO e o MÍNIMO
        return self.opt.pin_max_approval_thres < self.normalize(distance) > self.opt.pin_min_approval_thres

    def normalize(self, dist):
        # TODO: Normalizar o valor para uso do threshold
        return dist


def middleware_execution(detection):
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
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--pin_min_approval_thres", type=float, default=0.4,
                        help="pin min distance threshold for approval")
    parser.add_argument("--pin_max_approval_thres", type=float, default=0.4,
                        help="pin max distance threshold for approval")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, default="videos/fhd_c_ilum.mp4", help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--show_result", type=bool, default=False, help="after the detection, should display the image")
    parser.add_argument("--debug", type=bool, default=False, help="print debug messages in the execution")
    parser.add_argument("--runtime", type=bool, default=False, help="run as runtime - wait commands to execute")
    opt = parser.parse_args()

    # Starts detection object
    detection = Detect(opt)

    if opt.runtime:
        # Roda em modo de solicitação e resposta - integração com o middleware
        middleware_execution(detection)
    else:
        # Roda em modo de debug, de forma contínua
        detection.run("id", only_first_detection=False)
