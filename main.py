from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch.cuda
import numpy as np
import os

# Initialization
sam = sam_model_registry["vit_l"](checkpoint="./models/sam_vit_l_0b3195.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")


def is_outside_roi(roi_img: np.ndarray, x: int, y: int) -> bool:
    """
    Check if given cood are within the image region of interest

    :param roi_img: data image for reference on dimension
    :param x: x cood to check if in ROI
    :param y: y cood to check if in ROI
    :return: True if coordinates are not in region of interest
    """
    rh, rw, _ = roi_img.shape
    roi_x, roi_y = 200, 80

    return not (roi_x <= x < rw - roi_x and roi_y <= y < rh - roi_y)


def downscale(upscale_img: np.ndarray, resize_factor: int = 6) -> np.ndarray:
    """
    Resize an image smaller if feasible

    :param upscale_img: Original image to be processed
    :param resize_factor: A number to divide the original image size by. Default 6
    :return: Downscaled image
    """

    if upscale_img.shape[0] < 1000:
        return upscale_img

    return cv2.resize(upscale_img, ((upscale_img.shape[1] // resize_factor), upscale_img.shape[0] // resize_factor))


def stupid_ceiling_detector(input_img: np.ndarray, most_left_x_cood: int = 0) -> int:
    """
    Given an image, and the most left coordinate of the detected pallets, guess where the ceiling is.

    The algorithm guesses the ceiling by checking the pixels on side of the pallet, and if there is a major change in pixel from middle to the top,
    we assume there ceiling to side corner is there.

    :param input_img: Input image of the container layer
    :param most_left_x_cood: The most left x coordinates of the detected pallets
    :return: y coordinates of the estimate ceiling
    """
    kernel_size: int = 8

    # Kernel size 1 will look like this
    # Width would be kernel left + lernel right + center
    # 1  1  1
    # 1  1  1
    # 1  1  1
    if most_left_x_cood - (kernel_size + kernel_size + 1) < 50:
        return most_left_x_cood

    # Grayscale because color does not contribute to comparison
    grey_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    h, w = grey_img.shape

    lookup_started = False
    x_min = most_left_x_cood - 1 - (kernel_size * 2)
    last_block: np.ndarray = np.zeros((((kernel_size * 2) + 1), ((kernel_size * 2) + 1)), dtype=np.uint8)
    last_lookup_y_pos: int = h // 2

    est_container_ceiling_y: int = 10

    # While the last lookup position does not exceed the top of the image
    while last_lookup_y_pos - kernel_size > 0:
        new_block = grey_img[last_lookup_y_pos - kernel_size: last_lookup_y_pos + kernel_size + 1,
                    x_min:most_left_x_cood]

        # Silly trick just to skip comparison for the first time
        if not lookup_started:
            last_block = new_block
            lookup_started = True
            continue

        # If it is not the first lookup, do a comparison
        image_diff = np.sum(cv2.subtract(new_block, last_block, dtype=cv2.CV_64F))
        diff: int = abs(image_diff if image_diff is not None else 0)

        if diff > 1500:
            # Give a buffer
            est_container_ceiling_y = last_lookup_y_pos + 20

        # Move the lookup values
        last_lookup_y_pos = last_lookup_y_pos - kernel_size - kernel_size - 1
        last_block = new_block

    # Common sense
    # If the top ceiling is less than a certain % of the image, it cant be
    if est_container_ceiling_y > input_img.shape[0] * 0.25:
        print("ceiling logic")
        est_container_ceiling_y = 10

    return est_container_ceiling_y


def fill_rate_calculation(prompt_points: list[list[int, int]], segment_mask: np.ndarray, ori: np.ndarray) -> float:
    """
    Takes in a segmented image, based on the inital prompt points, do some cleanup and perform fill rate calculation.

    This function also runs a sub function to find the ceiling of the container

    :param prompt_points: List of multiple [x, y] list. Eg: [ [x, y], [x , y]...]
    :param segment_mask: Segmented image with highlighted region
    :param ori: Original image to draw over boxes and for reference
    :return:
    """
    thresh = cv2.threshold(segment_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # Container dimensions
    # For this calculation, we will take the max and min of the pallets found as an assumption first
    cx, cy, cw, ch = 999999, 9999999, 0, 0

    fill_rate_used = np.zeros(ori.shape)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 1000:
            continue

        if any(x < ppx < x + w and y < ppy < y + h for ppx, ppy in prompt_points):
            cv2.drawContours(fill_rate_used, [c], -1, (255, 255, 255), thickness=-1)

            cx = min(x, cx)
            cy = min(y, cy)
            cw = max(x + w, cw)
            ch = max(y + h, ch)

    tallest: int = stupid_ceiling_detector(ori, cx)

    cv2.rectangle(ori, (cx, tallest), (cw, ch), (0, 0, 255), 3)

    # Dilate to fill the gaps a little more
    cv2.dilate(fill_rate_used, np.ones((8, 8)), fill_rate_used)

    total_white = np.sum(fill_rate_used[tallest:ch, cx: cw] == 255)
    total_black = np.sum(fill_rate_used[tallest:ch, cx: cw] == 0)

    return round(total_white / (total_white + total_black), 2)


def prompt_segment(prompt_points: list[list[int, int]], segment_img: np.ndarray):
    """
    Given a list of data points, prompt SAM to segment segment_img based on data prompt.

    :param prompt_points: List of multiple [x, y] list. Eg: [ [x, y], [x , y]...]
    :param segment_img: Image for SAM to segment from
    :return: The segmented mask
    """

    predictor = SamPredictor(sam)

    input_point_nd = np.array(prompt_points, dtype=np.int32)
    input_label = np.ones(len(prompt_points), dtype=np.int32)

    predictor.set_image(segment_img)
    masks, scores, _ = predictor.predict(
        point_coords=input_point_nd,
        point_labels=input_label,
        multimask_output=False,
    )

    return masks[0]


def pallet_label_detector(layer_img: np.ndarray) -> list[list[int, int]]:
    lower_val = np.array([150, 150, 150], dtype=np.uint8)
    upper_val = np.array([255, 255, 255], dtype=np.uint8)

    # preparing the mask to overlay
    mask = cv2.inRange(layer_img, lower_val, upper_val)
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), mask)

    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # find contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # new_mask = np.ones(img.shape[:2], dtype="uint8") * 255

    prompt_points = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if is_outside_roi(layer_img, x, y):
            continue

        if w * h < 1000:
            continue

        # cv2.rectangle(new_mask, (x, y), (x + w, y + h), (0, 0, 255), -1)
        prompt_points.append([int(x + (w / 2)), int(y + (h / 2))])

    # res_final = cv2.bitwise_and(layer_img, layer_img, mask=cv2.bitwise_not(new_mask))

    return prompt_points


def process_fill_rate(img_fp: str):
    input_layer_img: np.ndarray = cv2.imread(img_fp)
    input_layer_img = downscale(input_layer_img)

    # First, find all the labels in the image
    # The label position can help prompt SAM to generate segments better
    label_points: list[list[int, int]] = pallet_label_detector(input_layer_img)

    # Send the labels position to SAM and get a segment mask
    segmented_mask: np.ndarray = prompt_segment(label_points, input_layer_img)

    # Draw on the original image with values from the mask
    segment_color = np.random.random(3) * 100

    segmented_img = input_layer_img.copy()
    segmented_img[segmented_mask] = segment_color
    mask = cv2.inRange(segmented_img, segment_color - 10, segment_color + 10)

    # Based on the segmented image, find the fill rate
    fill_rate: float = fill_rate_calculation(label_points, mask, segmented_img)

    # Display
    segmented_img = cv2.putText(segmented_img, img_fp, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                (146, 146, 208), 2, cv2.LINE_AA)
    segmented_img = cv2.putText(segmented_img, "Fill Rate: " + str(fill_rate), (50, 80), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                (146, 146, 208), 2, cv2.LINE_AA)
    ori_and_mask: np.ndarray = np.hstack((input_layer_img, segmented_img))
    ori_and_mask = cv2.resize(ori_and_mask, (int(ori_and_mask.shape[1] / 1.5), int(ori_and_mask.shape[0] / 1.5)))

    print(img_fp, fill_rate)
    cv2.imshow("out", ori_and_mask)


if __name__ == '__main__':

    image_dir = "./data"

    for filename in os.listdir(image_dir):
        f = os.path.join(image_dir, filename)

        # checking if it is a file
        if os.path.isfile(f):
            process_fill_rate(f)
            cv2.waitKey()
