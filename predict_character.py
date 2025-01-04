# import os
# import cv2
# import copy
# import numpy as np 
# from scipy import stats
# import keras

# Models = keras.models.load_model('modelocr.keras')



# label_dict = { 'Ain' : 'ع', 'alif' :'ا', 'ba' : 'ب', 'daad' : 'ض', 'dal' : 'د', 'fa' : 'ف', 'gem' : 'ج', 'gen' : 'غ', 'ha':'هـ', 'haa' :'ح', 'kaf':'ك', 
#               'khaa' : 'خ', 'lam' : 'ل', 'lam_alif' : 'لا', 'mim' : 'م', 'nun' : 'ن', 'qaf' : 'ق', 'raa' : 'ر', 'saad' : 'ص', 'shen' : 'ش', 'sin' : 'س', 
#               'taa' : 'ت', 'tah' : 'ط', 'thaa' : 'ث', 'waw' : 'و', 'yaa' : 'ي', 'zah' : 'ظ', 'zal' : 'ذ', 'zin' : 'ز'}

# def Predict(Characters, Evaluate = False):
#     Predictions = []
#     Model_Predictions = []

#     for Characters in Characters:
#         Prediction = []
#         Model_Prediction = []

#         for Character in Characters:
#             # gray = cv2.cvtColor(Character, cv2.COLOR_BGR2GRAY)

#             # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
#             # morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

#             # for i in range(morph.shape[0]):
#             #     for j in range(morph.shape[1]):
#             #         if not morph[i][j]:
#             #             morph[i][j] = 1
            
#             # div = gray / morph
#             # gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

#             _, thresh = cv2.threshold(Character, 180, 255, cv2.THRESH_BINARY_INV)
#             thresh = cv2.resize(thresh, (100, 100), interpolation = cv2.INTER_AREA)

#             x = np.array([thresh]).reshape(-1, 100, 100, 1) / 255.0
#             y = Models.predict(x)
#             y = np.argmax(y)
#             predicted_label = list(label_dict.keys())[y]
#             Model_Prediction.append(predicted_label)
#             Prediction.append(label_dict[predicted_label])
            

#         Predictions.append(copy.deepcopy(Prediction))
#         Model_Predictions.append(copy.deepcopy(Model_Prediction))

#     if Evaluate:
#         return Model_Predictions

#     return Predictions
import os
import cv2
import copy
import numpy as np 
from scipy import stats
import keras
# import matplotlib.pyplot as plt  # Import untuk menampilkan gambar

# Load model
Models = keras.models.load_model('modelOCRnew.keras')

# label_dict = { 
#     'Ain' : 'ع', 'alif' :'ا', 'ba' : 'ب', 'daad' : 'ض', 'dal' : 'د', 'fa' : 'ف', 
#     'gem' : 'ج', 'gen' : 'غ', 'ha':'ه', 'haa' :'ح', 'kaf':'ك', 'khaa' : 'خ', 
#     'lam' : 'ل', 'lam_alif' : 'لا', 'mim' : 'م', 'nun' : 'ن', 'qaf' : 'ق', 'raa' : 'ر', 
#     'saad' : 'ص', 'shen' : 'ش', 'sin' : 'س', 'taa' : 'ت', 'tah' : 'ط', 'thaa' : 'ث', 
#     'waw' : 'و', 'yaa' : 'ي', 'zah' : 'ظ', 'zal' : 'ذ', 'zin' : 'ز'
# }
label_dict = { 
    'Ain' : 'Ai', 'alif' :'A', 'ba' : 'B', 'daad' : 'Dh', 'dal' : 'D', 'fa' : 'F', 
    'gem' : 'J', 'gen' : 'Gh', 'ha':'H', 'haa' :'KH', 'kaf':'K', 'khaa' : 'KHA', 
    'lam' : 'L', 'lam_alif' : 'llf', 'mim' : 'M', 'nun' : 'N', 'qaf' : 'Q', 'raa' : 'R', 
    'saad' : 'Sh', 'shen' : 'Sy', 'sin' : 'S', 'taa' : 'T', 'tah' : 'Th', 'thaa' : 'Ts', 
    'waw' : 'W', 'yaa' : 'Y', 'zah' : 'Zh', 'zal' : 'Dz', 'zin' : 'Z'
}


def Predict(Characters, Evaluate=False):
    Predictions = []
    Model_Predictions = []

    for Characters in Characters:
        Prediction = []
        Model_Prediction = []

        for Character in Characters:
            # Apply thresholding and resizing to the character image
            gray = cv2.cvtColor(Character, cv2.COLOR_BGR2GRAY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            for i in range(morph.shape[0]):
                for j in range(morph.shape[1]):
                    if not morph[i][j]:
                        morph[i][j] = 1
            
            div = gray / morph
            gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            # thresh = cv2.resize(thresh, (100, 100), interpolation=cv2.INTER_AREA)

            original_h, original_w = thresh.shape
            target_size = 100
            scale = min(target_size / original_h, target_size / original_w)  # Scale to fit within target size
            
            new_h = int(original_h * scale)
            new_w = int(original_w * scale)
            resized = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)

            top = (target_size - new_h) // 2
            bottom = target_size - new_h - top
            left = (target_size - new_w) // 2
            right = target_size - new_w - left

            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


            # Show the image of the character being predicted
            # plt.imshow(padded, cmap='gray')
            # plt.title(f'Predicted Image')
            # plt.axis('off')  # Hide axes
            # plt.show()

            # Prepare the character image for prediction
            x = np.array([padded]).reshape(-1, 100, 100, 1) / 255.0
            y = Models.predict(x)
            y = np.argmax(y)
            predicted_label = list(label_dict.keys())[y]
            Model_Prediction.append(predicted_label)
            Prediction.append(label_dict[predicted_label])

        Predictions.append(copy.deepcopy(Prediction))
        Model_Predictions.append(copy.deepcopy(Model_Prediction))

    if Evaluate:
        return Model_Predictions

    return Predictions
      

