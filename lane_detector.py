
import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    return cv2.Canny(blur, 50, 150)


def region_of_interest(image, poi):
    polygons = np.array([poi])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def visualize_lines(frame, lines):
    line_img = np.zeros_like(frame)
    h, w, _ = frame.shape
    if lines is not None:
        for line in lines:
            if len(line) == 0:
                continue
            x1, y1, x2, y2 = line.reshape(-1)
            x1 = max(0, x1)
            x2 = max(0, x2)
            y1 = max(0, y1)
            y2 = max(0, y2)

            x1 = min(w, x1)
            x2 = min(w, x2)
            y1 = min(h, y1)
            y2 = min(h, y2)

            color = (255, 0, 0) # (0, 0, 255) if x1 < 230 and x2 < 230 else (255, 0, 0)
            cv2.line(line_img, (x1, y1), (x2, y2), color, 10)

    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combo_img


def detect_lines(frame, poi):
    canny_img = canny(frame)
    roi_img = region_of_interest(canny_img, poi)
    lines = cv2.HoughLinesP(roi_img,
                            rho=2,
                            theta=np.pi/180,
                            threshold=75,
                            lines=np.array([]),
                            minLineLength=10,
                            maxLineGap=50)

    return lines


def average_slope_intercept(frame, lines):
    # left_other = []  # Left line of whole road
    left = []
    right = []
    if lines is None:
        return np.array([])
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        params = np.polyfit((x1, x2), (y1, y2), deg=1)
        slope = params[0]
        y_intercept = params[1]

        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))

    # Average out all values for left and right into a single line
    if left:
        left_avg = np.average(left, axis=0)
    else:
        left_avg = np.array([])
    if right:
        right_avg = np.average(right, axis=0)
    else:
        right_avg = np.array([])

    # if left_other:
    #     left_other_avg = np.average(left_other, axis=0)
    # else:
    #     left_other_avg = np.array([])

    # Calculate end points for left and right line
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    # left_other_line = calculate_coordinates(frame, left_other_avg)

    return np.array([left_line, right_line])


def calculate_coordinates(frame, parameters):
    if len(parameters) == 0:
        return np.array([])
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1*(3/5))
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


name = 'road_video'
video_path = f"./data/{name}.mp4"

# Show how lines are detected on first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
lane_frame = np.copy(frame)
print(lane_frame.shape)
cv2.imshow('frame', frame)
cv2.waitKey(0)

canny_img = canny(lane_frame)
cv2.imshow('canny', canny_img)
cv2.waitKey(0)

h = lane_frame.shape[0]
if name == 'road_video':
    poi = [(250, h), (1100, h), (550, 300)]
elif name == 'driving_mumbai':
    poi = [(70, h), (700, h), (368, 190)]

roi_img = region_of_interest(canny_img, poi)
cv2.imshow('ROI', roi_img)
cv2.waitKey(0)

lines = cv2.HoughLinesP(roi_img,
                        rho=2,
                        theta=np.pi/180,
                        threshold=50,
                        lines=np.array([]),
                        minLineLength=10,
                        maxLineGap=50)

lines = average_slope_intercept(lane_frame, lines)
combo_img = visualize_lines(lane_frame, lines)
cv2.imshow('result', combo_img)
cv2.waitKey(0)

# Start tracking lines in video
cap = cv2.VideoCapture(video_path)
i = 0
while cap.isOpened():
    _, frame = cap.read()

    lines = detect_lines(frame, poi)
    avg_lines = average_slope_intercept(frame, lines)
    combo_img = visualize_lines(frame, avg_lines)
    cv2.imshow('Combo', combo_img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
