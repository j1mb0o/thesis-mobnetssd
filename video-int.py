import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='models/model-uint8v2.tflite')

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

        frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_to_rgb, size)
        frame_norm = frame_resized - 128
        data = np.expand_dims(frame_resized, axis=0).astype(np.int8)

        print(data.dtype)

        start = time.time()

        interpreter.set_tensor(input_index[0]['index'], data)
        interpreter.invoke()

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

        print(scores)

        for i in range(int(num_of_detections)):
            if scores[i] > 0:
                xmin = int(max(boxes[i][1] * frame.shape[1], 1))
                ymin = int(max(boxes[i][0] * frame.shape[0], 1))
                xmax = int(min(boxes[i][3] * frame.shape[1], frame.shape[1]))
                ymax = int(min(boxes[i][2] * frame.shape[0], frame.shape[0]))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), class_color[abs(int(classes[i]))], 2)

        end = time.time()

        print('time elapsed: ' + str(end - start) + 's')

        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
