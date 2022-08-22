from os import path
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='mobilenet.tflite')

data = np.ndarray(shape=(1, 320, 320, 3), dtype=np.float32)

class_color = [(255, 0, 0),      #blue   bicycle
               (0, 255, 0),      #green  bus
               (0, 0, 255),      #red    car
               (255, 255, 0),    #cyan   motorcycle
               (255, 0, 255),    #purple pedestrian
               (0, 255, 255)]    #yellow truck


size = (320, 320)
# original_image = cv2.imread('pic2.jpg')
original_image = cv2.imread('aachen_000013_000019_leftImg8bit.png')

image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_norm = cv2.resize(original_image, size)

image = (image_norm / 127.5) - 1

data[0] = image


start = time.time()

interpreter.allocate_tensors()
input_index = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_index[0]['index'], data)

interpreter.invoke()

scores = interpreter.get_tensor(output_details[0]['index'])
boxes = interpreter.get_tensor(output_details[1]['index'])
num_of_detections = interpreter.get_tensor(output_details[2]['index'])
classes = interpreter.get_tensor(output_details[3]['index'])

print(boxes[0][0][0])
print(image.shape[0])
print(f'{boxes[0]}')
print(f'{scores[0]}')
print(f'{classes[0]}')
print(f'{num_of_detections[0]}')

# exit(0)

for i in range(int(num_of_detections[0])):
    # if scores[0][i] > 0.5:

    xmin = int(boxes[0][i][1] * original_image.shape[1])
    ymin = int(boxes[0][i][0] * original_image.shape[0])
    xmax = int(boxes[0][i][3] * original_image.shape[1])
    ymax = int(boxes[0][i][2] * original_image.shape[0])

    cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), class_color[int(classes[0][i])], 2)

end = time.time()

print('time elapsed: ' + str(end-start)+'s')
# exit(0)

# cv2.imshow('sample', cv2.resize(original_image, (1000, 1000)))
cv2.imshow('sample', original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
