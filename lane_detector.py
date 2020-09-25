
import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image, (5, 5), sigmaX=0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    h = image.shape[0]
    polygons = np.array([
        [(70, h), (700, h), (368, 200)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def visualize_lines(frame, lines):
    line_img = np.zeros_like(frame)
    h, w = frame.shape
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(-1)
            print(x1, y1)
            print(x2, y2)
            x1 = max(0, x1)
            x1 = min(w, x1)
            y1 = max(0, y1)
            y1 = min(h, y1)
            x2 = max(0, x2)
            x2 = min(w, x2)
            y2 = max(0, y2)
            y2 = min(h, y2)

            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)

    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combo_img


def detect_lines(frame):
    canny_img = canny(frame)
    roi_img = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(roi_img,
                            rho=2,
                            theta=np.pi/180,
                            threshold=30,
                            lines=np.array([]),
                            minLineLength=30,
                            maxLineGap=50)

    return lines


def filter_lines(frame, lines):
    left = []
    right = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        if abs(angle) > 60:
            continue

        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        y_intercept = params[1]

        # Check for horisontal lines
        # if abs(y2 - y1) < 5:
        #     print('yes')
        #     continue

        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))

    # Average out all values for left and right into a single line
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)

    # Calculate end points for left and right line
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)

    return np.array([left_line, right_line])


def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


video_path = f"./data/driving_mumbai.mp4"

# Show how lines are detected on first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
lane_frame = cv2.cvtColor(np.copy(frame), cv2.COLOR_RGB2GRAY)
cv2.imshow('frame', lane_frame)
cv2.waitKey(0)

canny_img = canny(lane_frame)
cv2.imshow('canny', canny_img)
cv2.waitKey(0)

roi_img = region_of_interest(canny_img)
cv2.imshow('ROI', roi_img)
cv2.waitKey(0)

lines = cv2.HoughLinesP(roi_img,
                        rho=2,
                        theta=np.pi/180,
                        threshold=50,
                        lines=np.array([]),
                        minLineLength=30,
                        maxLineGap=50)

lines = filter_lines(lane_frame, lines)
combo_img = visualize_lines(lane_frame, lines)
cv2.imshow('combo_img', combo_img)
cv2.waitKey(0)

# Start tracking lines in video
cap = cv2.VideoCapture(video_path)
while True:
    ret, orig_frame = cap.read()
    frame = orig_frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if ret is None:
        break

    lines = detect_lines(frame)
    lines = filter_lines(frame, lines)
    combo_img = visualize_lines(frame, lines)
    cv2.imshow('Combo img', combo_img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


