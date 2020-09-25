
import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
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


def overlay_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(-1)
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)

    combo_img = cv2.addWeighted(image, 0.8, line_img, 1, 1)
    return combo_img


def detect_lines(img):
    canny_img = canny(img)
    roi_img = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(roi_img,
                        rho=1,
                        theta=np.pi/180,
                        threshold=50,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=50)

    return lines


video_path = f"./data/driving_mumbai.mp4"

# Show how lines are detected on first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
lane_frame = np.copy(frame)
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

combo_img = overlay_lines(frame, lines)
cv2.imshow('combo_img', combo_img)
cv2.waitKey(0)

# Start tracking lines in video
cap = cv2.VideoCapture(video_path)
while True:
    ret, orig_frame = cap.read()
    frame = orig_frame.copy()

    if ret is None:
        break

    lines = detect_lines(frame)
    combo_img = overlay_lines(frame, lines)
    cv2.imshow('Combo img', combo_img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


