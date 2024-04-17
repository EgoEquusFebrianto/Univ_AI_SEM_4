import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("D:/ML/rockpaperscissors_model.h5")

##img_path = "D:/ML/rockpaperscissors/scissors/0ePX1wuCc3et7leL.png" # percobaan 1
##img_path = "D:/ML/_______.png" # percobaan 2
##img_path = "D:/ML/images.jpg" # percobaan 3
##img_path = "D:/ML/images.png" # percobaan 4
##img_path = "D:/ML/images-lSYUAIcFt-transformed.png" # percobaan 5
##img_path = "D:/ML/preview_cut.png" # percobaan 6
##img_path = "D:/ML/HOHO.png" # percobaan 7
##img_path = "D:/ML/HOHO-removebg-preview.png" # percobaan 8
##img_path = "D:/ML/HOHO-removebg-preview-transformed.png" # percobaan 9
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalisasi

prediction = model.predict(img_array)

predicted_class_index = np.argmax(prediction)

# Cursor
class_labels = {0: 'rock', 1: 'paper', 2: 'scissors'}
predicted_class_label = class_labels[predicted_class_index]

print(f"The predicted class is: {predicted_class_label}")

"""
catatan :
semua yang di data set pasti benar diprediksi.
yang gagal adalah percobaan ke: 3, 4, 7
Pertanyaannya adalah 'WHY ?'
Bagi yang mau mengembangkan algo.. bisa bisa aja ya teeheee...
kalau udah di upload ya ke tempat yang bisa di download oke oce....
"""
