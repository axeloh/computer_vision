
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import time
import configparser
from collections import defaultdict
import json

def normalize(x, a, b):
    # return 255 * x / np.max(x)
    return (b-a) * (x - np.min(x)) / (np.max(x) - np.min(x) + 1e8) + a

def gaussian_blur(img):
    g = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ])
    return (1 / 273) * signal.convolve(img, g, mode='same')


def spatial_grad_mag(img, Gx, Gy, threshold=None):
    """Approximates first derivative in x and y direction"""
    start = time.time()
    rows = img.shape[0]
    columns = img.shape[1]
    mag = np.zeros_like(img)

    for i in range(1, rows - 2):
        for j in range(1, columns - 2):
            s1 = np.sum(np.sum(img[i:i + 3, j:j + 3] * Gx, axis=0))
            s2 = np.sum(np.sum(img[i:i + 3, j:j + 3] * Gy, axis=1))
            # s1 = np.sum(np.sum((np.array([[1],[2],[1]]) @ np.array([[-1, 0, 1]])) * img[i:i+3, j:j+3], axis=0))
            # s2 = np.sum(np.sum((np.array([[-1],[0],[1]]) @ np.array([[1, 2, 1]])) * img[i:i+3, j:j+3], axis=1))
            # s1 = np.sum(np.sum(signal.convolve2d(img[i:i+3, j:j+3], Gx), axis=1))
            # s2 = np.sum(np.sum(signal.convolve2d(img[i:i+3, j:j+3], Gy), axis=0))

            mag[i + 1][j + 1] = np.sqrt(s1 ** 2 + s2 ** 2)

    # Normalize
    # mag = 255.0 * mag / np.max(mag)

    if threshold:
        threshold = 70  # varies for application [0 255]
        mag[mag < threshold] = threshold
        mag[mag == round(threshold)] = 0

    print(f'Done in {(time.time() - start):.1f}s')
    return mag

def time_grad(img1, img2):
    """Approximates the first derivative wrt time """
    n = 10
    mask1 = np.zeros((n, n))
    mask1.fill(-1)
    mask2 = np.ones((n,n))

    rows = img1.shape[0]
    cols = img1.shape[1]

    return -1 * img1 + img2

    mag = np.zeros_like(img1)
    for i in range(1, rows - n):
        for j in range(1, cols - n):
            s1 = np.sum(np.sum(img1[i:i + n, j:j + n] * mask1))
            s2 = np.sum(np.sum(img2[i:i + n, j:j + n] * mask2))

            mag[i + 1][j + 1] = np.sqrt(s1 ** 2 + s2 ** 2)

    print(mag.shape)
    # Normalize
    # mag = 255.0 * mag / np.max(mag)

    return mag

def laplacian(img):
    """ Approximates the second derivative in x and y direction,
     using a laplacian mask
     Returns f - f_avg ≈ f_xx + f_yy
     """

    mask = np.array([
        [0, -1/4, 0],
        [-1/4, 1, -1/4],
        [0, -1/4, 0]])

    rows = img.shape[0]
    cols = img.shape[1]

    mag = np.zeros_like(img)
    for i in range(1, rows - 2):
        for j in range(1, cols - 2):
            mag[i + 1][j + 1] = np.sum(np.sum(img[i:i + 3, j:j + 3] * mask, axis=0))

    # Normalize
    # mag = 255.0 * mag / np.max(mag)
    # mag = normalize(mag, 0, 255)
    return img - mag

video_name = 'car_pov'
video_path = f"../../data/{video_name}.mp4"
cmap = 'gray'

# Get config for video
with open('./config.json') as json_file:
    data = json.load(json_file)[video_name]
    skip_frames = data['skip_frames']
    window_w = data['window_w']
    window_h = data['window_h']
    dense_params = data['flow_params']


# cv2.WINDOW_NORMAL makes the output window resizealbe
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)

# resize the window according to the screen resolution
# cv2.resizeWindow('Resized Window', 1400, 600) # drone1
cv2.resizeWindow('Resized Window', window_w, window_h) # car_pov

# Start
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Sobel Filters
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]) # /8
sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]) # /8


# sobel_x = np.array([
#     [1, 2, 0, -2, -1],
#     [4, 8, 0, -8, 4],
#     [6, 12, 0, -12, -6],
#     [4, 8, 0, -8, 4],
#     [1, 2, 0, -2, -1]
# ])

# sobel_x = np.array([
#     [-2, -1, 0, 1, 2],
#     [-2, -1, 0, 1, 2],
#     [-2, -1, 0, 1, 2],
#     [-2, -1, 0, 1, 2],
#     [-2, -1, 0, 1, 2]
# ])

#sobel_y = sobel_x.T

# --------------------------------------------------------------------------------
# Calculate and display gradients
# --------------------------------------------------------------------------------
prev = frame
while True:
    ret, orig_frame = cap.read()
    for _ in range(skip_frames):
        ret, frame = cap.read()

    if ret is None:
        print("No more frames.")
        break

    gray_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    frame = gaussian_blur(gray_frame)

    # Compute gradient wrt x, y, and t
    gx = signal.convolve2d(frame, sobel_x, mode='same')
    gy = signal.convolve2d(frame, sobel_y, mode='same')
    # frame = normalize(frame, 0, 255)

    # Time gradient
    # Mask image 1: [-1, -1, -1, -1], mask image 2: [1, 1, 1, 1]
    gt = time_grad(prev, frame)
    #gt = frame - prev

    gx = cv2.convertScaleAbs(gx)
    gy = cv2.convertScaleAbs(gy)
    gt = cv2.convertScaleAbs(gt)
    # frame = cv2.convertScaleAbs(frame)

    concat = np.concatenate((gray_frame, gt, gx, gy), axis=1)  # axis=1 for horisontal concat
    cv2.imshow('Resized Window', concat)
    #cv2.imshow('flow', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame)
        #cv2.imwrite('opticalhsv.png', bgr)
    prev = frame


# --------------------------------------------------------------------------------
# Calculate optical flow (Horn & Chunck)
# --------------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame = gaussian_blur(frame)

# Initialize (u, v)
u = np.zeros_like(frame)
v = np.zeros_like(frame)
l = 0.1  # lambda

# For visualization
hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
hsv[..., 1] = 255

direction_arrow_params = dict(
    color=(0,0,0),
    thickness=5,
    tipLength=0.7
)


prev = frame
while True:
    ret, orig_frame = cap.read()
    for _ in range(skip_frames):
        ret, frame = cap.read()

    if ret is None:
        print("No more frames.")
        break

    frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    #frame = gaussian_blur(frame)
    # blurred = cv2.GaussianBlur(frame, ksize=(11,11), sigmaX=0)
    blurred = frame

    # ----------- Own version -----------
    # # Compute gradient wrt x, y, and t
    # gx = signal.convolve2d(blurred, sobel_x, mode='same')
    # gy = signal.convolve2d(blurred, sobel_y, mode='same')
    # gt = blurred - prev
    #
    # # Apply Laplace function
    # mean_u = cv2.Laplacian(u, ddepth=cv2.CV_16S, ksize=3)
    # #mean_u = laplacian(u)
    # # converting back to uint8
    # #mean_v = laplacian(v)
    # # mean_u = cv2.convertScaleAbs(mean_u)
    #
    # mean_v = cv2.Laplacian(v, ddepth=cv2.CV_16S, ksize=3)
    # #mean_v = cv2.convertScaleAbs(mean_v)
    #
    # # Update, p, d, u and v
    # p = gx * mean_u + gy * mean_v + gt
    # d = l + gx**2 + gy**2
    #
    # u = mean_u - gx * p/d
    # v = mean_v - gy * p/d
    #
    # u = cv2.blur(u, ksize=(15,15))
    # v = cv2.blur(v, ksize=(15,15))
    #
    # # Uncomment to visualize flow
    # mag, ang = cv2.cartToPolar(u, v)
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 0] = cv2.normalize(hsv[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ----------- OpenCV version -----------
    flow = cv2.calcOpticalFlowFarneback(prev, frame, None, **dense_params)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang  * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ----------- Motion direction estimation -----------
    # Predict whether robot is moving left or right
    thresh = 1
    #hor_motion = u
    hor_motion = flow[..., 0]
    print(np.mean(hor_motion))

    objects_to_right = hor_motion[hor_motion > thresh]
    objects_to_left = hor_motion[hor_motion < -thresh]
    objects_straight = hor_motion[(hor_motion > -thresh) & (hor_motion < thresh)]

    robot_left_motion = np.count_nonzero(objects_to_right)
    robot_right_motion = np.count_nonzero(objects_to_left)
    robot_straight_motion = np.count_nonzero(objects_straight)
    # print(f'{robot_left_motion} | {robot_straight_motion} | {robot_right_motion}')

    mid = orig_frame.shape[1] // 2
    motion_direction = np.argmax([robot_left_motion, robot_right_motion, robot_straight_motion])

    if motion_direction == 0:
        # Left motion
        label = 'left'
        arrow_start = mid + 50
        arrow_end = mid - 50
        cv2.arrowedLine(orig_frame, (arrow_start, int(0.8 * orig_frame.shape[0])),
                        (arrow_end, int(0.8 * orig_frame.shape[0])), **direction_arrow_params)
    elif motion_direction == 1:
        # Right motion
        label = 'right'
        arrow_start = mid - 50
        arrow_end = mid + 50
        cv2.arrowedLine(orig_frame, (arrow_start, int(0.8 * orig_frame.shape[0])),
                        (arrow_end, int(0.8 * orig_frame.shape[0])), **direction_arrow_params)
    else:
        # Straight (or other) motion
        label = 'straight'
        arrow_start = int(0.8 * orig_frame.shape[0]) + 50
        arrow_end = int(0.8 * orig_frame.shape[0]) - 50
        cv2.arrowedLine(orig_frame, (mid, arrow_start),
                        (mid, arrow_end), **direction_arrow_params)

    #cv2.putText(orig_frame, label, (mid, int(0.95 * orig_frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX,
     #           0.5, (0, 0, 0), 1)


    # ----------- Show origin frame + flow -----------
    concat = np.concatenate((orig_frame, bgr), axis=1)  # axis=1 for horisontal concat
    cv2.imshow('Resized Window', concat)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame)
        cv2.imwrite('opticalhsv.png', bgr)

    # Update
    u = cv2.convertScaleAbs(u)
    v = cv2.convertScaleAbs(v)
    prev = frame