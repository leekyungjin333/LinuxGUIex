# 0507.py
import cv2
import numpy as np

src = np.array([[0, 0, 0, 0],
                [1, 1, 3, 5],
                [6, 1, 1, 3],
                [4, 3, 1, 7]
              ], dtype=np.uint8)

dst = cv2.equalizeHist(src)
print('dst =', dst)
