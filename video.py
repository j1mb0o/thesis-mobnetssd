from time import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# interpreter = tflite.Interpreter(model_path='mobilenet.tflite')
interpreter = tflite.Interpreter(model_path='models/model-float16.tflite')
data = np.ndarray(shape=(1, 320, 320, 3), dtype=np.float32)

class_color = [(255, 0, 0),      #blue   bicycle
               (0, 255, 0),      #green  bus
               (0, 0, 255),      #red    car
               (255, 255, 0),    #cyan   motorcycle
               (255, 0, 255),    #purple pedestrian
               (0, 255, 255)]    #yellow truck


size = (320, 320)

interpreter.allocate_tensors()
input_index = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture('videos/video.mp4')

while cap.isOpened():

    ret, frame = cap.read()

    if ret:
        start = time()

        frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resize = image_norm = cv2.resize(frame_to_rgb, size)
        frame_norm = (frame_resize / 127.5) - 1

        data[0] = frame_norm

        interpreter.set_tensor(input_index[0]['index'], data)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        num_of_detections = interpreter.get_tensor(output_details[2]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]

        for i in range(int(num_of_detections)):
            if scores[i] > 0.50:

                xmin = int(boxes[i][1] * frame.shape[1])
                ymin = int(boxes[i][0] * frame.shape[0])
                xmax = int(boxes[i][3] * frame.shape[1])
                ymax = int(boxes[i][2] * frame.shape[0])

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), class_color[int(classes[i])], 2)

        cv2.imshow('Frame', frame)
        end = time()
        print(f'time elapsed: {round(end - start, 2)}. FPS: {int(1 / (end - start))}')

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
