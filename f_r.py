import numpy as np
import cv2



def extract_minutiae(img):
    # 对图像进行预处理
    img = cv2.medianBlur(img, 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 获取图像的梯度图像
    gradient_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # 寻找Minutiae点
    minutiae = []
    for i in range(1, gradient.shape[0] - 1):
        for j in range(1, gradient.shape[1] - 1):
            pixel = gradient[i, j]
            if pixel == 0:
                continue
            local_max = True
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if gradient[i + x, j + y] > pixel:
                        local_max = False
                        break
                if not local_max:
                    break
            if local_max:
                minutiae.append((i, j))
    return minutiae

def compare_fingerprints(minutiae1, minutiae2):
    # 计算两个Minutiae点的相似度
    score = 0
    for m1 in minutiae1:
        for m2 in minutiae2:
            d = np.sqrt(np.square(m1[0] - m2[0]) + np.square(m1[1] - m2[1]))
            if d < 20:
                score += 1
    return score/max(len(minutiae1), len(minutiae2))
