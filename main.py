
import os
import cv2

import split_words
import split_character
import predict_character


Path = 'output'
Images = sorted(os.listdir(Path))
# image_path = "Words/no-brand_no-brand_full011.jpg"

for Image_Name in Images:
    Words =split_words.Split_Words(cv2.imread(os.path.join(Path, Image_Name)))
    # Words =split_words.Split_Words(cv2.imread(image))
    Characters = split_character.Split_Characters(Words)
    Predictions = predict_character.Predict(Characters)

    Words = []
    for Prediction in Predictions:
        Word = ''.join(Prediction)
        Words.append(Word)
    Words = ' '.join(Words)

    print(Words)



    


