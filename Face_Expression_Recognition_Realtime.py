# Importing necessary libraries
import tensorflow as tf
import numpy as np
import cv2
import dlib
import pickle

# get_model() function
def get_model():
    backbone = tf.keras.applications.EfficientNetB2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=None
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model

# Loading the trained model
model = get_model()
model.load_weights("best_weights.h5") 

# Load LabelEncoder 
def load_object(name):
    pickle_obj = open(f"{name}.pck","rb")
    obj = pickle.load(pickle_obj)
    return obj

Le = load_object("LabelEncoder")

# Helper functions
def ProcessImage(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image , [96 , 96] , method="bilinear")
    image = tf.expand_dims(image , 0)
    return image

def RealtimePrediction(image , model, encoder_):
    prediction = model.predict(image)
    prediction = np.argmax(prediction , axis = 1)
    return encoder_.inverse_transform(prediction)[0]

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Video capturing and real-time prediction loop
VideoCapture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

while True :
    
    ret , frame = VideoCapture.read() 
    
    if not ret :
        break

    gray = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY)

    rects = detector(gray , 0)

    if len(rects) >= 1 :
        for rect in rects :
            (x , y , w , h) = rect_to_bb(rect)
            img = gray[y-10 : y+h+10 , x-10 : x+w+10]
            
            if img.shape[0] == 0 or img.shape[1] == 0 :
                cv2.imshow("Frame", frame)
                
            else :
                img = cv2.cvtColor(img , cv2.COLOR_GRAY2RGB)
                img = ProcessImage(img)
                out = RealtimePrediction(img , model , Le)
                cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
                z = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, str(out), (x, z), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                
        cv2.imshow("Frame", frame)
            
    else :
        cv2.imshow("Frame", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
VideoCapture.release()
cv2.destroyAllWindows()

