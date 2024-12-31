import cv2
import numpy as np

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
        result = Word.copy()
        image_bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > average_area:
                label = 'Large'
            else:
                label = 'Small'
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
        small_boxes = [box for box in image_bounding_boxes if box["classification"] == "Small"]
        large_boxes = [box for box in image_bounding_boxes if box["classification"] == "Large"]
        grouped_boxes = []
        merged_boxes = []
        visited_indices = set()
        classified_boxes = []
        used_boxes = set()
        large_box_indicators = {}

        for idx, large_box in enumerate(large_boxes):
            large_box_indicators[(large_box["x"], large_box["w"])] = idx + 1  
        for i in range(len(small_boxes)):
            if i in visited_indices:
                continue
            current_group = [small_boxes[i]]
            visited_indices.add(i)
            has_merged = True
            while has_merged:
                has_merged = False
                for j in range(len(small_boxes)):
                    if j in visited_indices:
                        continue
                    for box in current_group:
                        box2 = small_boxes[j]
                        if (box["x"] < box2["x"] + box2["w"] and box["x"] + box["w"] > box2["x"]):
                            current_group.append(box2)
                            visited_indices.add(j)
                            has_merged = True
                            break 
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
                if (merged_box["x"] <= large_box["x"] + large_box["w"] and
                    merged_box["x"] + merged_box["w"] >= large_box["x"]):
                    group["contained_boxes"].append(merged_box)
            if group["contained_boxes"]:
                grouped_boxes.append(group)
        for group in grouped_boxes:
            if len(group["contained_boxes"]) == 1:
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
                for box in group["contained_boxes"]:
                    large_box = group["large_box"]
                    distance_to_xw = abs((box["x"] + box["w"]) - (large_box["x"] + large_box["w"]))
                    distance_to_x = abs(box["x"] - large_box["x"])
                    if distance_to_xw < min_distance_to_xw:
                        min_distance_to_xw = distance_to_xw
                        closest_to_xw = box
                    if distance_to_x < min_distance_to_x:
                        min_distance_to_x = distance_to_x
                        closest_to_x = box
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
            if key not in unique_boxes or any([box["awal"], box["tengah"], box["akhir"]]):
                unique_boxes[key] = box
        final_classified_boxes = list(unique_boxes.values())
        for classified_box in final_classified_boxes:
            x = classified_box['x']
            w = classified_box['w']
            awal = classified_box['awal']
            akhir = classified_box['akhir']
            
            for group in grouped_boxes:
                large_box = group['large_box']
                if (x >= large_box['x'] and x + w <= large_box['x'] + large_box['w']):
                    if awal and not akhir:
                        classified_box['w'] = (large_box['x'] + large_box['w']) - x
                    elif akhir and not awal:
                        classified_box['x'] = large_box['x']
                        classified_box['w'] = (x + w) - large_box['x']
                    elif awal and akhir:
                        classified_box['x'] = large_box['x']
                        classified_box['w'] = large_box['w']
                    break
        for group in grouped_boxes:
            large_box_w = group['large_box']['w']
            classified_boxes_in_group = [
                box for box in final_classified_boxes
                if (box['x'] >= group['large_box']['x'] and 
                    box['x'] + box['w'] <= group['large_box']['x'] + group['large_box']['w'] and
                    not (box['awal'] == 'first' and box['akhir'] == 'last'))
            ]
            
            total_w_classified_boxes = sum(box['w'] for box in classified_boxes_in_group)
            
            if total_w_classified_boxes > 0:
                remaining_w = large_box_w - total_w_classified_boxes
                if len(classified_boxes_in_group) > 0:
                    remaining_w_per_box = round(remaining_w / len(classified_boxes_in_group))
                    
                    if len(classified_boxes_in_group) == 2:
                        box1, box2 = classified_boxes_in_group
                        if box1['awal'] and box2['akhir']:
                            box1['x'] = box1['x'] - remaining_w_per_box  
                            box1['w'] = box1['w'] + (2 * remaining_w_per_box)  
                            box2['w'] = box2['w'] + remaining_w_per_box  

                    else:
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
                                half_remaining_w = round(remaining_w_per_box / 2)
                                box['x'] = x - half_remaining_w
                                box['w'] = w + (2 * half_remaining_w)
                    classified_boxes_in_group.sort(key=lambda b: b['x'])
                    for i in range(1, len(classified_boxes_in_group)):
                        prev_box = classified_boxes_in_group[i - 1]
                        curr_box = classified_boxes_in_group[i]
                        if curr_box['x'] < prev_box['x'] + prev_box['w']:
                            overlap = (prev_box['x'] + prev_box['w']) - curr_box['x']
                            curr_box['x'] += overlap
                            curr_box['w'] -= overlap
        final_classified_boxes.sort(key=lambda box: box['x'])

        for classified_box in final_classified_boxes:
            x = classified_box['x']
            w = classified_box['w']
            cropped_image = result[:, x:x+w]
            Characters.append(cropped_image)  
    return Characters
