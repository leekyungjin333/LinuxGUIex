# 0919.py
import cv2

src1 = cv2.imread('./data/stitch_image1.jpg')
src2 = cv2.imread('./data/stitch_image2.jpg')
src3 = cv2.imread('./data/stitch_image3.jpg')
src4 = cv2.imread('./data/stitch_image4.jpg')

stitcher = cv2.createStitcher()
status, dst = stitcher.stitch((src1, src2, src3, src4))
cv2.imwrite('./data/stitch_out.jpg', dst)
cv2.imshow('dst',  dst)
cv2.waitKey()
cv2.destroyAllWindows()
