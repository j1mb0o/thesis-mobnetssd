from os import path
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class_color = [(255, 0, 0),      #blue   bicycle
               (0, 255, 0),      #green  bus
               (0, 0, 255),      #red    car
               (255, 255, 0),    #cyan   motorcycle
               (255, 0, 255),    #purple pedestrian
               (0, 255, 255)]    #yellow truck


size = (320, 320)
original_image = cv2.imread('images/G0048711_JPG.rf.ab7b940485ceda6d8f3127489b80f7d5.jpg')
image_to_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_to_rgb, size)


def float_inf():
    data = np.ndarray(shape=(1, 320, 320, 3), dtype=np.float32)
    interpreter = tflite.Interpreter(model_path='models/model-float16.tflite')
    image_norm = (image_resized / 127.5) - 1

    data[0] = image_norm

    start = time.time()

    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_index[0]['index'], data)

    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    num_of_detections = interpreter.get_tensor(output_details[2]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]

    print(f'{scores}')

    print(f'{boxes}')
    print(f'{classes}')
    print(f'{num_of_detections}')

    # exit(0)
    print(scores[-3], classes[-3])

    xmin = int(max(boxes[-3][1] * original_image.shape[1], 1))
    ymin = int(max(boxes[-3][0] * original_image.shape[0], 1))
    xmax = int(min(boxes[-3][3] * original_image.shape[1], original_image.shape[1]))
    ymax = int(min(boxes[-3][2] * original_image.shape[0], original_image.shape[0]))

    cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), class_color[abs(int(classes[-3]))], 2)
    # for i in range(int(num_of_detections[0])):
    #     # if scores[i] > 0.45:
    #
    #     xmin = int(boxes[i][1] * original_image.shape[1])
    #     ymin = int(boxes[i][0] * original_image.shape[0])
    #     xmax = int(boxes[i][3] * original_image.shape[1])
    #     ymax = int(boxes[i][2] * original_image.shape[0])
    #
    #     cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), class_color[int(classes[i])], 2)

    end = time.time()

    print('time elapsed: ' + str(end-start)+'s')
    # exit(0)

    cv2.imshow('sample', cv2.resize(original_image, (1500, 1000)))
    # cv2.imshow('sample', original_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def int_inf():

    interpreter = tflite.Interpreter(model_path='models/model-uint8v2.tflite')

    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_quant, input_zero = input_index[0]['quantization']
    print(input_quant, input_zero)
    # print(test['dtype'])
    # print(test['quantization'])
    # print(test)
    # exit(0)
    data = np.ndarray(shape=(1, 320, 320, 3), dtype=np.int8)
    image_norm = image_resized - 128
    # image_norm = input_quant * (image_resized - input_zero)

    data[0] = image_norm

    start = time.time()

    interpreter.set_tensor(input_index[0]['index'], data)

    # test = interpreter.get_input_details()[0]
    # print(test['dtype'])
    # print(test['quantization'])
    # print(test)
    # exit(0)

    interpreter.invoke()
    print(interpreter.get_output_details()[0])
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    num_of_detections = interpreter.get_tensor(output_details[2]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]

    scores_quant, scores_zero = output_details[0]['quantization']
    boxes_quant, boxes_zero = output_details[1]['quantization']
    num_quant, num_zero = output_details[2]['quantization']
    classes_quant, classes_zero = output_details[3]['quantization']
    #
    scores = (scores_quant * (scores - scores_zero))
    boxes = boxes_quant * (boxes - boxes_zero)
    classes = classes_quant * (classes - classes_zero)
    num_of_detections = num_quant * (num_of_detections - num_zero)

    print(f'{boxes=}')
    print(f'{scores=}')
    print(f'{classes=}')
    print(f'{num_of_detections=}')

    # exit(0)
    print(scores[-3], classes[-3])

    xmin = int(max(boxes[-3][1] * original_image.shape[1], 1))
    ymin = int(max(boxes[-3][0] * original_image.shape[0], 1))
    xmax = int(min(boxes[-3][3] * original_image.shape[1], original_image.shape[1]))
    ymax = int(min(boxes[-3][2] * original_image.shape[0], original_image.shape[0]))

    cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), class_color[abs(int(classes[-3]))], 2)
    # for i in range(int(num_of_detections)):
    #     if scores[i] > 0:
    #
    #         xmin = int(max(boxes[i][1] * original_image.shape[1], 1))
    #         ymin = int(max(boxes[i][0] * original_image.shape[0], 1))
    #         xmax = int(min(boxes[i][3] * original_image.shape[1], original_image.shape[1]))
    #         ymax = int(min(boxes[i][2] * original_image.shape[0], original_image.shape[0]))
    #
    #         cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), class_color[abs(int(classes[i]))], 2)

    end = time.time()
    #
    print('time elapsed: ' + str(end-start)+'s')

    cv2.imshow('sample', cv2.resize(original_image, (1500, 1000)))
    # cv2.imshow('sample', original_image)

    cv2.waitKey(0)
    # time.sleep(5)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    float_inf()
    # int_inf()

