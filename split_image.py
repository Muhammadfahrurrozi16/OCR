import cv2
import numpy as np

def process_image(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image,150,255,cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for idx,contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contour)
        bounding_boxes.append((idx,x,y,w,h))
    bounding_boxes.sort(key=lambda b: b[2])

    merged_boxes = []
    current_row = []

    for bbox in bounding_boxes:
        idx,x,y,w,h = bbox
        if not current_row:
            current_row.append(bbox)
        else:
            _, _, prev_y, _, prev_h = current_row[-1]
            if y <= prev_y + prev_h:
                current_row.append(bbox)
            else:
                min_x = min([b[1] for b in current_row])
                max_x = max([b[1] + b[3] for b in current_row])
                min_y = min([b[2] for b in current_row])
                max_y = max([b[2] + b[4] for b in current_row])
                merged_boxes.append((min_x,min_y,max_x - min_x,max_y - min_y))
                current_row = [bbox]
    if current_row:
        min_x = min([b[1] for b in current_row])
        max_x = max([b[1] + b[3] for b in current_row])
        min_y = min([b[2] for b in current_row])
        max_y = max([b[2] + b[4] for b in current_row])
        merged_boxes.append((min_x,min_y,max_x - min_x,max_y - min_y))
    if merged_boxes:
        avg_height = np.mean([bbox[3] for bbox in merged_boxes])
    else:
        avg_height = 0
    labeled_boxes = []
    large_boxes = []
    small_boxes = []

    for (x,y,w,h) in merged_boxes:
        label = "large" if h > avg_height else "small"
        labeled_boxes.append((x,y,w,h,label))
        if label == "large":
            large_boxes.append((x,y,w,h))
        else:
            small_boxes.append((x,y,w,h))
    counts = []

    for lbox in large_boxes:
        lx, ly, lw, lh = lbox
        small_count = 0
        large_count = 0

        for box in bounding_boxes:
            _, bx,by,bw,bh = box
            if lx <= bx and by >= ly and (bx + bw) <= (lx + lw) and (by + bh) <= (ly + lh):
                if bh > avg_height:
                    large_count += 1
                else: 
                    small_count += 1
        counts.append((small_count,large_count))
    final_boxes = []
    for lbox in large_boxes:
        lx,ly,lw,lh = lbox
        sub_boxes = [lbox]

        for box in labeled_boxes:
            bx,by,bw,bh,label = box
            if lx <= bx and by >= ly and (bx + bw) <= (lx + lw) and (by + bh) <= (ly + lh):
                sub_boxes.append((bx,by,bw,bh))
        min_x = min([b[0] for b in sub_boxes])
        max_x = max([b[0] + b[2] for b in sub_boxes])
        min_y = min([b[1] for b in sub_boxes])
        max_y = max([b[1] + b[3] for b in sub_boxes])
        final_boxes.append((min_x,min_y,max_x - min_x,max_y - min_y))
    large_boxes = [(box,count) for box,count in zip(final_boxes,counts) if  box[3] > avg_height]

    if large_boxes:
        max_large_count = max(count[1] for _, count in large_boxes)
        max_large_boxes = [box for box,count in large_boxes if count[1] == max_large_count]
        if len(max_large_boxes) == 1:
            Lines = max_large_boxes[0]
        else:
            Lines = min(max_large_boxes, key=lambda box: counts[(final_boxes.index(box))][0])
    else:
        Lines = None
    return Lines