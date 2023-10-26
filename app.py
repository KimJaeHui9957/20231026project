from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__, template_folder="templates")

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    if image:
        image_path = 'static/images/uploaded_image.jpg'
        image.save(image_path)

        # YOLOv4로 이미지 분석
        net = cv2.dnn.readNetFromDarknet('yolo/yolov4.cfg', 'yolo/yolov4.weights')
        with open('yolo/coco.names', 'r') as f:
            classes = f.read().strip().split('\n')

        img = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        output_layers = net.getUnconnectedOutLayersNames()
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])).astype(int)
                    x, y = center_x - w // 2, center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        result = ""
        for i in range(len(boxes)):
            if i in indexes:
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                result += f"{label}: {confidence:.2f}\n"

        return render_template('upload.html', result=result, image_path=image_path)
    return "No image provided."

if __name__ == '__main__':
    app.run(debug=True)