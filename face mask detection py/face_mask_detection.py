from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as num
import cv2 as cv





def detect(image, model="mask_detector.model", cnf=0.5):
    protoText = "deploy.prototxt.txt"
    weights = "res10_300x300_ssd_iter_140000.caffemodel"
    Net = cv.dnn.readNet(protoText, weights)

    model = load_model(model)

    image = cv.imread(image)

    (h, w) = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    Net.setInput(blob)
    detections = Net.forward()

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf > cnf:
            box = detections[0, 0, i, 3:7] * num.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


            try:
                face = image[startY:endY, startX:endX]
                face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                face = cv.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = num.expand_dims(face, axis=0)

                (mask, withoutMask) = model.predict(face)[0]
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (255, 0, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv.putText(image, label, (startX, startY - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv.rectangle(image, (startX, startY), (endX, endY), color, 1)
            except:
                pass

    return image


if __name__ == "__main__":
    path=input("enter image path : \n")
    result = detect(image=path)
    cv.imshow("output", result)
    print("showing result")
    cv.waitKey(0)

