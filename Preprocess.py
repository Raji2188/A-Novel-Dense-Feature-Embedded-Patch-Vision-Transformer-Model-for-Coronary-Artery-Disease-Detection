import cv2

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def GABF(img):
    gaussian_blur = cv2.GaussianBlur(img, (5,5), 0)
    bilateral_gaussian = cv2.bilateralFilter(gaussian_blur, 15, 75, 75)
    CLAHE_image = apply_clahe(bilateral_gaussian)
    return CLAHE_image
