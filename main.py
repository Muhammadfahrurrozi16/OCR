import os
import pandas as pd
import split_words
import split_character
import predict_character


Path = 'output'
Images = sorted(os.listdir(Path))
file_path = "hadistlist.csv"
df = pd.read_csv(file_path)


def hitung_kemiripan(row, target_list, target_length):
    match_count = 0  
    row_values = row["Kalimat latin"].split(",") 
    row_values = [huruf.strip() for huruf in row_values if huruf]  
    row_length = len(row_values)

    for huruf in row_values:
        if huruf in target_list:
            match_count += 1

    length_similarity = 1 / (1 + abs(row_length - target_length))  
    total_score = match_count + length_similarity
    return total_score
def get_api_url(image):
    Words = split_words.Split_Words(image)
    Characters = split_character.Split_Characters(Words)
    Predictions = predict_character.Predict(Characters)

    Words = []
    for Prediction in Predictions:
        Word = ','.join(Prediction)
        Words.append(Word)
    Words.reverse()
    Words = ','.join(Words)
    kumpulan_huruf_list = [huruf for huruf in Words.split(",") if huruf]
    panjang_target = len(kumpulan_huruf_list)
    df["similarity_score"] = df.apply(
        lambda row: hitung_kemiripan(row, kumpulan_huruf_list, panjang_target), axis=1
    )
    baris_terbaik = df.loc[df["similarity_score"].idxmax()]
    api_url = baris_terbaik["api"]
    return api_url

    


