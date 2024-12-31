# from textwrap import wrap
import cv2
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
def Sorting_Key(rect):
    global Lines, Size

    x, y, w, h = rect

    cx = x + int(w / 2)
    cy = y + int(h / 2)

    for i, (upper, lower) in enumerate(Lines):
        if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
            return cx + ((i + 1) * Size)

def Split_Words(Image):
    global Lines, Size

    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    for i in range(morph.shape[0]):
        for j in range(morph.shape[1]):
            if not morph[i][j]:
                morph[i][j] = 1
    
    div = gray / morph
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)

    i = 0
    Length = len(contours)
    while i < Length:
        x, y, w, h = cv2.boundingRect(contours[i])

        if w * h <= 100:
            del contours[i]
            i -= 1
            Length -= 1
        i += 1
    
    h_proj = np.sum(thresh, axis = 1)

    upper = None
    lower = None
    Lines = []
    for i in range(h_proj.shape[0]):
        proj = h_proj[i]

        if proj != 0 and upper == None:
            upper = i
        elif proj == 0 and upper != None and lower == None:
            lower = i
            if lower - upper >= 30:
                Lines.append([upper, lower])
            upper = None
            lower = None

    if upper:
        Lines.append([upper, h_proj.shape[0] - 1])

    Size = thresh.shape[1]

    bounding_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        for upper, lower in Lines:
            if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
                bounding_rects.append([x, y, w, h])

    i = 0  
    Length = len(bounding_rects) 
    while i < Length:
        x, y, w, h = bounding_rects[i]
        j = 0

        while j < Length:
            distancex = abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2]))
            distancey = abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3]))

            threshx = max(abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2])),
                          abs(bounding_rects[j][0] - bounding_rects[i][0]),
                          abs((bounding_rects[j][0] + bounding_rects[j][2]) - bounding_rects[i][0]),
                          abs((bounding_rects[j][0] + bounding_rects[j][2]) - (bounding_rects[i][0] + bounding_rects[i][2])))

            threshy = max(abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3])),
                          abs(bounding_rects[j][1] - bounding_rects[i][1]),
                          abs((bounding_rects[j][1] + bounding_rects[j][3]) - bounding_rects[i][1]),
                          abs((bounding_rects[j][1] + bounding_rects[j][3]) - (bounding_rects[i][1] + bounding_rects[i][3])))

            if i != j and any([all([not any([all([bounding_rects[j][1] > y + h, bounding_rects[j][1] + bounding_rects[j][3] > y + h]), all([bounding_rects[j][1] < y, bounding_rects[j][1] + bounding_rects[j][3] < y])]),
                                   not any([all([bounding_rects[j][0] > x + w, bounding_rects[j][0] + bounding_rects[j][2] > x + w]), all([bounding_rects[j][0] < x, bounding_rects[j][0] + bounding_rects[j][2] < x])])]),
                              all([distancex <= 10, bounding_rects[i][3] + bounding_rects[j][3] + 10 >= threshy]), all([bounding_rects[i][2] + bounding_rects[j][2] + 10 >= threshx, distancey <= 10])]):
                
                x = min(bounding_rects[i][0], bounding_rects[j][0])
                w = max(bounding_rects[i][0] + bounding_rects[i][2], bounding_rects[j][0] + bounding_rects[j][2]) - x
                y = min(bounding_rects[i][1], bounding_rects[j][1])
                h = max(bounding_rects[i][1] + bounding_rects[i][3], bounding_rects[j][1] + bounding_rects[j][3]) - y

                bounding_rects[i] = [x, y, w, h]
                del bounding_rects[j]
                i = -1
                Length -= 1
                break

            j += 1
        i += 1

    bounding_rects.sort(key = Sorting_Key)

    Words = []
    for x, y, w, h in bounding_rects:
        crop = Image[y:y + h, x:x+ w]
        Words.append(crop.copy())
    # print("Bounding boxes (w, h) for each character:")
    # for x, y, w, h in bounding_rects:
    #     print(f"Width: {w}, Height: {h}")

    return Words


def Split_Characters(Words):
    Characters = []
    average_areas_per_image = []
    bounding_boxes_data = []

    for Word in Words:
        gray = cv2.cvtColor(Word, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        for i in range(morph.shape[0]):
            for j in range(morph.shape[1]):
                if not morph[i][j]:
                    morph[i][j] = 1
        
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

        _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = list(contours)
        
        # Gambar bounding box
        areas = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            areas.append(area)

        if areas:
            average_area = sum(areas) / len(areas)
        else:
            average_area = 0
        
        average_areas_per_image.append(average_area)
        
        # Klasifikasi dan gambar bounding box
        result = Word.copy()
        image_bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Klasifikasi sebagai 'small' atau 'large'
            if area > average_area:
                label = 'Large'
                color = (0, 0, 255)  # Merah
            else:
                label = 'Small'
                color = (0, 255, 0)  # Hijau
            # print(f"Small Bounding Box - x: {x}, y: {y}, w: {w}, h: {h}, area: {area}")
            # cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(result, f"{label}: {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, color, 2)
            # Gambar bounding box
            bounding_box_info = {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "classification": label
            }
            image_bounding_boxes.append(bounding_box_info)
        
        bounding_boxes_data.append({
            "image_id": len(bounding_boxes_data) + 1,
            "bounding_boxes": image_bounding_boxes
        })

        # Cek tumpang tindih pada bounding box 'Small'
        small_boxes = [box for box in image_bounding_boxes if box["classification"] == "Small"]
        large_boxes = [box for box in image_bounding_boxes if box["classification"] == "Large"]
        grouped_boxes = []
        merged_boxes = []
        visited_indices = set()
        classified_boxes = []
        used_boxes = set()
        large_box_indicators = {}

        for idx, large_box in enumerate(large_boxes):
            large_box_indicators[(large_box["x"], large_box["w"])] = idx + 1  # Indeks mulai dari 1
 

        for i in range(len(small_boxes)):
            if i in visited_indices:
                continue

            # Mulai dengan satu bounding box
            current_group = [small_boxes[i]]
            visited_indices.add(i)

            # Cari semua bounding box yang tumpang tindih dengan grup ini
            has_merged = True
            while has_merged:
                has_merged = False
                for j in range(len(small_boxes)):
                    if j in visited_indices:
                        continue

                    for box in current_group:
                        box2 = small_boxes[j]
                        if (box["x"] < box2["x"] + box2["w"] and box["x"] + box["w"] > box2["x"]):
                            # print(f"Overlapping detected: {box} with {box2}")
                            current_group.append(box2)
                            visited_indices.add(j)
                            has_merged = True
                            break  # Hentikan loop untuk `box` dan mulai lagi untuk `box2`

            # Gabungkan semua bounding box dalam grup ini
            combined_x = min(box["x"] for box in current_group)
            combined_w = max(box["x"] + box["w"] for box in current_group) - combined_x

            merged_boxes.append({"x": combined_x, "w": combined_w})
        for i in range(len(small_boxes)):
            if i not in visited_indices:
                merged_boxes.append(small_boxes[i])
        for large_box in large_boxes:
            group = {
                "large_box": large_box,
                "contained_boxes": [],
                "indicator": large_box_indicators[(large_box["x"], large_box["w"])]
            }

            for merged_box in merged_boxes:
                # Cek apakah merged_box berada dalam large_box
                if (merged_box["x"] <= large_box["x"] + large_box["w"] and
                    merged_box["x"] + merged_box["w"] >= large_box["x"]):
                    group["contained_boxes"].append(merged_box)
            
            if group["contained_boxes"]:
                grouped_boxes.append(group)
        for group in grouped_boxes:
            if len(group["contained_boxes"]) == 1:
                # Jika hanya ada satu merged_box, klasifikasikan sebagai both 'first' dan 'last'
                box = group["contained_boxes"][0]
                classified_boxes.append({
                    "x": box["x"],
                    "w": box["w"],
                    "awal": "first",
                    "tengah": None,
                    "akhir": "last",
                    "indicator": group["indicator"]
                })
                used_boxes.add((box["x"], box["w"]))
            else:
                closest_to_xw = None
                closest_to_x = None
                min_distance_to_xw = float('inf')
                min_distance_to_x = float('inf')

                # Tentukan bounding box yang paling dekat dengan nilai x+w dan x
                for box in group["contained_boxes"]:
                    large_box = group["large_box"]

                    # Jarak ke nilai x + w dari large_box
                    distance_to_xw = abs((box["x"] + box["w"]) - (large_box["x"] + large_box["w"]))
                    # Jarak ke nilai x dari large_box
                    distance_to_x = abs(box["x"] - large_box["x"])

                    # Update jika ditemukan jarak yang lebih dekat untuk x+w
                    if distance_to_xw < min_distance_to_xw:
                        min_distance_to_xw = distance_to_xw
                        closest_to_xw = box

                    # Update jika ditemukan jarak yang lebih dekat untuk x
                    if distance_to_x < min_distance_to_x:
                        min_distance_to_x = distance_to_x
                        closest_to_x = box

                # Klasifikasikan bounding boxes
                for box in group["contained_boxes"]:
                    if (box["x"], box["w"]) not in used_boxes:
                        if box == closest_to_xw:
                            classified_boxes.append({
                                "x": box["x"],
                                "w": box["w"],
                                "awal": "first",
                                "tengah": None,
                                "akhir": None,
                                "indicator": group["indicator"]
                            })
                        elif box == closest_to_x:
                            classified_boxes.append({
                                "x": box["x"],
                                "w": box["w"],
                                "awal": None,
                                "tengah": None,
                                "akhir": "last",
                                "indicator": group["indicator"]
                            })
                        else:
                            classified_boxes.append({
                                "x": box["x"],
                                "w": box["w"],
                                "awal": None,
                                "tengah": "middle",
                                "akhir": None,
                                "indicator": group["indicator"]
                            })
                    used_boxes.add((box["x"], box["w"]))
            for box in merged_boxes:
                if (box["x"], box["w"]) not in used_boxes:
                    classified_boxes.append({
                        "x": box["x"],
                        "w": box["w"],
                        "awal": None,
                        "tengah": None,
                        "akhir": None,
                        "indicator": None
                        })
                    used_boxes.add((box["x"], box["w"]))
        final_classified_boxes = []
        unique_boxes = {}

        for box in classified_boxes:
            key = (box["x"], box["w"])
            
            # Jika key belum ada atau box baru memiliki atribut klasifikasi, gantikan
            if key not in unique_boxes or any([box["awal"], box["tengah"], box["akhir"]]):
                unique_boxes[key] = box

        # Simpan hasil akhir tanpa duplikasi
        final_classified_boxes = list(unique_boxes.values())
        print("Before adjustment:")
        for classified_box in final_classified_boxes:
            print(f"Box: x={classified_box['x']}, w={classified_box['w']}, "
                f"awal={classified_box['awal']}, tengah={classified_box['tengah']}, akhir={classified_box['akhir']}")

        # Memperbarui w pada classified_box berdasarkan kondisi awal dan akhir
        for classified_box in final_classified_boxes:
            x = classified_box['x']
            w = classified_box['w']
            awal = classified_box['awal']
            akhir = classified_box['akhir']
            
            for group in grouped_boxes:
                large_box = group['large_box']
                
                # Cek apakah classified_box berada dalam rentang large_box
                if (x >= large_box['x'] and x + w <= large_box['x'] + large_box['w']):
                    if awal and not akhir:
                        # Jika hanya atribut awal yang terisi
                        classified_box['w'] = (large_box['x'] + large_box['w']) - x
                    elif akhir and not awal:
                        # Jika hanya atribut akhir yang terisi
                        classified_box['x'] = large_box['x']
                        classified_box['w'] = (x + w) - large_box['x']
                    elif awal and akhir:
                        # Jika atribut awal dan akhir terisi
                        classified_box['x'] = large_box['x']
                        classified_box['w'] = large_box['w']
                    break

        # Menghitung dan menambahkan remaining_w_per_box pada tiap classified_box dalam grup
        for group in grouped_boxes:
            large_box_w = group['large_box']['w']
            
            # Filter untuk mendapatkan hanya classified_boxes yang ada dalam large_box dan bukan 'first' dan 'last'
            classified_boxes_in_group = [
                box for box in final_classified_boxes
                if (box['x'] >= group['large_box']['x'] and 
                    box['x'] + box['w'] <= group['large_box']['x'] + group['large_box']['w'] and
                    not (box['awal'] == 'first' and box['akhir'] == 'last'))
            ]
            
            total_w_classified_boxes = sum(box['w'] for box in classified_boxes_in_group)
            
            if total_w_classified_boxes > 0:
                remaining_w = large_box_w - total_w_classified_boxes
                
                # Menghitung remaining_w per box dan membulatkan hasilnya
                if len(classified_boxes_in_group) > 0:
                    remaining_w_per_box = round(remaining_w / len(classified_boxes_in_group))
                    
                    if len(classified_boxes_in_group) == 2:
                        box1, box2 = classified_boxes_in_group
                        
                        if box1['awal'] and box2['akhir']:
                            # Jika box1 adalah awal dan box2 adalah akhir
                            # Penyesuaian untuk box awal
                            box1['x'] = box1['x'] - remaining_w_per_box  # Menggeser box awal ke kiri
                            box1['w'] = box1['w'] + (2 * remaining_w_per_box)  # Menambah lebar box awal
                            
                            # Penyesuaian untuk box akhir
                            box2['w'] = box2['w'] + remaining_w_per_box  # Menambah lebar box akhir

                    else:
                    # Terapkan penyesuaian pada setiap box dalam grup
                        for box in classified_boxes_in_group:
                            x = box['x']
                            w = box['w']
                            awal = box['awal']
                            akhir = box['akhir']
                            tengah = box['tengah']
                            
                            if awal and not akhir and not tengah:
                                # Jika hanya atribut awal yang terisi
                                box['x'] = x - remaining_w_per_box
                                box['w'] = w + (2 * remaining_w_per_box)
                                
                            elif akhir and not awal and not tengah:
                                # Jika hanya atribut akhir yang terisi
                                box['w'] = w + remaining_w_per_box

                            elif tengah and not awal and not akhir:
                                # Jika hanya atribut tengah yang terisi
                                half_remaining_w = round(remaining_w_per_box / 2)
                                box['x'] = x - half_remaining_w
                                box['w'] = w + (2 * half_remaining_w)
                        
                    # Pastikan tidak terjadi overlapping
                    classified_boxes_in_group.sort(key=lambda b: b['x'])
                    for i in range(1, len(classified_boxes_in_group)):
                        prev_box = classified_boxes_in_group[i - 1]
                        curr_box = classified_boxes_in_group[i]
                        if curr_box['x'] < prev_box['x'] + prev_box['w']:
                            overlap = (prev_box['x'] + prev_box['w']) - curr_box['x']
                            curr_box['x'] += overlap
                            curr_box['w'] -= overlap

        # Menampilkan hasil setelah proses penyesuaian
        print("\nAfter adjustment:")
        for classified_box in final_classified_boxes:
            print(f"Updated Box: x={classified_box['x']}, w={classified_box['w']}, "
                f"awal={classified_box['awal']}, tengah={classified_box['tengah']}, akhir={classified_box['akhir']}")
        final_classified_boxes.sort(key=lambda box: box['x'])

        for classified_box in final_classified_boxes:
            x = classified_box['x']
            w = classified_box['w']
            
            cropped_image = result[:, x:x+w]
            
            cv2.imwrite(f"cropped_{x}_{w}.png", cropped_image)
            
            Characters.append(cropped_image)  # Simpan gambar yang sudah di-crop dalam list Characters

    return Characters

    #     for classified_box in final_classified_boxes:
    #         x = classified_box['x']
    #         w = classified_box['w']
            
    #         # Crop gambar berdasarkan koordinat x dan w
    #         cropped_image = result[:, x:x+w]

    #     # Characters.append(cropped_images)
    #     cv2.imshow(f"Cropped Image: x={x}, w={w}",  cropped_image)
    # # cv2.imwrite(f"cropped_{x}_{w}.png", cropped_image)

    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # return Characters

        #     print(f"Large Bounding Box: {group['large_box']}")
        #     for contained_box in group["contained_boxes"]:
        #         print(f" - Contained Merged Box: {contained_box}")
        # print("\nFinal list of bounding boxes (x, y, w, h):")
        # for box in merged_boxes:
        #     print(f"x: {box['x']}, w: {box['w']}")

        # for box in  final_classified_boxes:
        #     # Potong gambar berdasarkan bounding box yang sudah digabungkan jika ada overlap
        #     cropped_image = Word[:, box["x"]:box["x"] + box["w"]]
            
        #     # Simpan atau proses hasil potongan sesuai kebutuhan
        #     Characters.append(cropped_image)
            
        #     # Jika Anda ingin menyimpan gambar, gunakan cv2.imwrite()
        #     image_id = len(bounding_boxes_data)
        #     output_filename = f"image_{image_id}_box_{merged_boxes.index(box)}.png"
        #     cv2.imwrite(output_filename, cropped_image)
                
    #             # Jika Anda ingin menyimpan gambar, gunakan cv2.imwrite()
    #             # Misal, simpan gambar potongan dengan nama berdasarkan id gambar dan index bounding box
            
    #     # Tampilkan hasil dengan bounding box
        # show_image(result, f'Bounding Boxes with Classification')
    # return Characters
    # # Tampilkan rata-rata luas bounding box per gambar
    # for i, avg_area in enumerate(average_areas_per_image):
    #     print(f'Rata-rata luas bounding box untuk gambar ke-{i+1}: {avg_area}')
    
        # Tampilkan hasil dengan bounding box
        # show_image(result, 'Bounding Boxes')

    # Hitung rata-rata luas
            
    #     Word_Characters = []
    #     for x, y, w, h,label in image_bounding_boxes:
    #         new_x = max(0, x - 3)
    #         new_w = min(Word.shape[1] - new_x, w + (x - new_x) + 3)
            
    #         crop = original_thresh[y:y + h, new_x:new_x + new_w]
            
    #         h_proj = np.sum(crop, axis = 1)

    #         padding = None
    #         for i in range(h_proj.shape[0]):
    #             proj = h_proj[i]
                
    #             if proj != 0:
    #                 padding = i
    #                 break
            
    #         new_y = padding
    #         new_h = min(Word.shape[0] - new_y, h + new_y + 3)

    #         size = max(new_w, new_h)
    #         Character = np.zeros((size, size, 3), np.uint8)
    #         Character.fill(255)

    #         Character[int((size - new_h) / 2):int((size + new_h) / 2), int((size - new_w) / 2):int((size + new_w) / 2)] = Word[new_y:new_y + new_h, new_x:new_x + new_w]
    #         Word_Characters.append(Character.copy())

    #     Characters.append(copy.deepcopy(Word_Characters))

    # return Characters


# def display_image(title, image):
#     """Helper function to display an image using matplotlib"""
#     plt.figure(figsize=(8, 8))
#     if len(image.shape) == 3:
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     else:
#         plt.imshow(image, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
# def show_image(img, title='Image'):
#     plt.figure(figsize=(10, 5))
#     if len(img.shape) == 3:
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     else:
#         plt.imshow(img, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

def main():
    # Baca gambar input
    image = cv2.imread('46444415_2275434705863981_5572823077055627264_n.png')

    # Pisahkan kata-kata dari gambar
    words = Split_Words(image)

    # Tampilkan hasil ekstraksi kata
    # for i, word in enumerate(words):
    #     display_image(f'Word {i+1}', word)
    
    # Pisahkan karakter dari setiap kata
    Characters = Split_Characters(words)

# Menampilkan hasil crop
    # for i, character in enumerate(Characters):
    #     cv2.imshow(f"Character {i+1}", character)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # # Tampilkan hasil ekstraksi karakter untuk setiap kata
    # for i, char in enumerate(Characters):
    #     # for j, char in enumerate(word_characters):
    #         display_image(f'Character from Word {i+1}', char)

    # Atau gunakan cv2.imshow untuk menampilkan (optional, jika menjalankan dalam lingkungan yang mendukung GUI)
    # for i, word in enumerate(words):
    #     cv2.imshow(f'Word {i+1}', word)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # for i, word_characters in enumerate(characters):
    #     for j, char in enumerate(word_characters):
    #         cv2.imshow(f'Character {j+1} from Word {i+1}', char)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
