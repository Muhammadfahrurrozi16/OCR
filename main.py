import os
import pandas as pd
import split_words
import split_character
import predict_character


Path = 'output'
Images = sorted(os.listdir(Path))
file_path = "hadistnew.csv"
df = pd.read_csv(file_path)
image = "output/Snapinsta.app_269803409_626147785294798_8158947981556276484_n_1080.jpg"

def hitung_kemiripan(row, target_list, target_length):
    match_count = 0  # Jumlah kecocokan langsung
    row_values = [str(row[kolom]) for kolom in row.index if kolom != "api"]
    row_length = len(row_values)

    # Hitung jumlah huruf yang cocok
    for huruf in row_values:
        if huruf in target_list:
            match_count += 1

    # Hitung skor jarak panjang (lebih pendek lebih baik)
    length_similarity = 1 / (1 + abs(row_length - target_length))  # Semakin kecil selisih, semakin besar nilai

    # Gabungkan skor (kombinasi kecocokan dan panjang)
    total_score = match_count + length_similarity
    return total_score
def get_api_url(image):
    # Words =split_words.Split_Words(cv2.imread(os.path.join(Path, Image_Name)))
    Words = split_words.Split_Words(image)
    Characters = split_character.Split_Characters(Words)
    Predictions = predict_character.Predict(Characters)

    Words = []
    for Prediction in Predictions:
        Word = ' '.join(Prediction)
        Words.append(Word)
    Words = ' '.join(Words)
    print(Words)
    kumpulan_huruf_list = Words.split( )
    panjang_target = len(kumpulan_huruf_list)

    df["similarity_score"] = df.apply(lambda row: hitung_kemiripan(row, kumpulan_huruf_list, panjang_target), axis=1)
    baris_terbaik = df.loc[df["similarity_score"].idxmax()]
    api_url = baris_terbaik["api"]
    return api_url

    # try:
    #     response = requests.get(api_url)
    #     response.raise_for_status()  # Pastikan tidak ada error dalam request
    #     api_data = response.json()  # Jika respon berbentuk JSON
    # except requests.exceptions.RequestException as e:
    #     api_data = f"Error saat mengakses API: {e}"

    # # Output hasil
    # print("Baris terbaik API:", api_url)
    # print("Respon API:", api_data)
    # hasil_api = baris_terbaik["api"]
    # print("Baris terbaik API:", hasil_api)



    


