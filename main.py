import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pytesseract

def main():
    img = cv.imread('./images/placas2.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"

    kernel = np.ones((25, 25), np.float32) / 625
    dst = cv.filter2D(img, -1, kernel)
    blur = cv.GaussianBlur(img, (25, 25), 0)

    plt.subplot(1, 3, 1), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(blur), plt.title('Gaussian')
    plt.xticks([]), plt.yticks([])
    plt.show()


    img = cv.imread('./images/placas2.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray_img = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_img, 50, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)

    plt.imshow(cleaned, cmap='gray'), plt.title('Black Letters Only')
    plt.xticks([]), plt.yticks([])
    plt.show()

    laplacian = cv.Laplacian(cleaned, cv.CV_64F)
    sobelx = cv.Sobel(cleaned, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(cleaned, cv.CV_64F, 0, 1, ksize=5)

    plt.subplot(2, 2, 1), plt.imshow(cleaned, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()



    # Output dtype = cv.CV_8U
    sobelx8u = cv.Sobel(cleaned, cv.CV_8U, 1, 0, ksize=5)

    # Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
    sobelx64f = cv.Sobel(cleaned, cv.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    plt.subplot(1, 3, 1), plt.imshow(cleaned, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap='gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap='gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

    plt.show()

    text_detected = pytesseract.image_to_string(cleaned, config='--psm 6')
    print("Detected text:", text_detected)



if __name__ == '__main__':
    main()