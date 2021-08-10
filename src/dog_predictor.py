import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.python.ops.gen_array_ops import space_to_depth
from dog_face_detector import DogFaceDetector

dfd = DogFaceDetector()
model = load_model('../models/finalized_modelAdam.h5')
# model2 = load_model('../models/finalized_model2.h5')
sdog_model = load_model('../models/finalized_model_sdog.h5')
sdog_classes = pickle.load(open('../data/sdog_classes.pkl','rb'))

def predict_dog(img_path, age_model=model, sdog_model=sdog_model):
    age, age_pct = predict_dog_age(img_path, age_model)
    breed, breed_pct = predict_dog_breed(img_path, sdog_model)
    if age == 'Puppy':
        if breed[0].lower() in 'aeiou':
            prediction = f"This is an {breed} [{breed_pct}%] {age} [{age_pct}%]"
        else:
            prediction = f"This is a {breed} [{breed_pct}%] {age} [{age_pct}%]"
    elif age == 'Adult':
        prediction = f"This is an {age} [{age_pct}%] {breed} [{breed_pct}%]"
    else:
        prediction = f"This is a {age} [{age_pct}%] {breed} [{breed_pct}%]"
    print(prediction)
    return prediction

def predict_dog_age(img_path, model):
    dogage_classes = {0: 'Adult', 1: 'Puppy', 2: 'Senior'}
    dogface = dfd.get_dogface(img_path, save=False)[0]
    dogface = dogface * 1./255
    # print(dogface * 1./255)
    dogface = np.expand_dims(dogface, axis=0)
    prediction = model.predict(dogface)
    predicted_class = dogage_classes[np.argmax(prediction)]
    predicted_class_pct = np.round(100*np.max(prediction), 2)
    # cv2.putText(dfd.img_result, f"{predicted_class}: {predicted_class_pct}%", tuple(dfd.bbox_dims[0][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    # cv2.putText()
    # print(dfd.bbox_dims)
    # print(predicted_class, predicted_class_pct)
    # plt.imshow(dfd.img_result)
    return predicted_class, predicted_class_pct

def predict_dog_breed(img_path, model):
    img = img_to_array(load_img(img_path, target_size=(224,224))) * 1./255
    prediction = model.predict(np.expand_dims(img, axis=0))
    sort_idx = np.argsort(prediction, axis=-1)[0][::-1]
    sort_pct = np.sort(prediction, axis=-1)[0][::-1]
    breed = sdog_classes[sort_idx[0]]
    breed_pct = round(sort_pct[0]*100, 2)
    return breed, breed_pct


if __name__ == "__main__": 
    img_path = '../img/yanna1.jpeg'
    # dfd = DogFaceDetector()

    # img_path = '../../data/Stanford/Images/n02109047-Great_Dane/n02109047_481.jpg'
    # print(predict_dog_breed(img_path))
    predict_dog(img_path)
    # plt.show()
    # sdog_model.predict(cv2.resize(cv2.imread(img_path), dsize=(224,224)))
    # img = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), dsize=(224,224))
    # predict_dog_age(img_path, model)
    # plt.show()
    # img = img_to_array(load_img(img_path, target_size=(224,224))) * 1./225
    # pred = sdog_model.predict(np.expand_dims(img, axis=0))
    # arg = np.argsort(pred, axis=-1)[0]
    # pct = np.sort(pred, axis=-1)[0]
    # print(arg[:-5:-1], pct[:-5:-1])
    # img.resize(224,224,3)

