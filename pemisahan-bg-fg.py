import numpy as np
import cv2

cap = cv2.VideoCapture("test.mp4")
if not cap.isOpened():
    raise RuntimeError("Video gagal dibuka. Path / codec bermasalah.")

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

cap.release()

frames = np.array(frames)      # (T, H, W)
T, H, W = frames.shape

X = frames.reshape(T, H * W)   # (T, HW)
X = X.astype(np.float64)

U, S, VT = np.linalg.svd(X, full_matrices=False)

k = 1

U_k  = U[:, :k]
S_k  = S[:k]
VT_k = VT[:k, :]

X_bg = U_k @ np.diag(S_k) @ VT_k

X_fg = X - X_bg

background = X_bg.reshape(T, H, W)
foreground = X_fg.reshape(T, H, W)

for i in range(0, T, 10):
    cv2.imshow("Original", frames[i])
    cv2.imshow("Background",
               np.clip(background[i], 0, 255).astype(np.uint8))
    cv2.imshow("Foreground",
               np.clip(np.abs(foreground[i]), 0, 255).astype(np.uint8))
    cv2.waitKey(200)

cv2.destroyAllWindows()

fps = 30  # atau ambil dari video asli
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out_bg = cv2.VideoWriter(
    "background_svdKANJUT.mp4", fourcc, fps, (W, H), isColor=False
)

out_fg = cv2.VideoWriter(
    "foreground_svdKANJUT.mp4", fourcc, fps, (W, H), isColor=False
)

for i in range(T):
    bg_frame = np.clip(background[i], 0, 255).astype(np.uint8)
    fg_frame = np.clip(np.abs(foreground[i]), 0, 255).astype(np.uint8)

    out_bg.write(bg_frame)
    out_fg.write(fg_frame)

out_bg.release()
out_fg.release()