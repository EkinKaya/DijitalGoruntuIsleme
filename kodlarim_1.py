import time 
import cv2
import numpy as np

start_time = time.time()

image = cv2.imread("01112v.jpg")
print(type(image))
print(image.shape)
print(image)

# Kenarlıkları kaldırmak için fonksiyon
def remove_edges(img, edge_size):
    return img[edge_size:-edge_size, edge_size:-edge_size, :]

# Kontrast ayarı yapmak için fonksiyon
def adjust_contrast(img, alpha, beta):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Histogram eşitleme yapmak için fonksiyon
def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cropped_image_1 = image[0:341,0:400,:]
cropped_image_1 = remove_edges(cropped_image_1, 20)
cropped_image_1 = adjust_contrast(cropped_image_1, 1.0, 0)
cropped_image_1 = equalize_histogram(cropped_image_1)

cropped_image_2 = image[341:682,0:400,:]
cropped_image_2 = remove_edges(cropped_image_2, 20)
cropped_image_2 = adjust_contrast(cropped_image_2, 1.0, 0)
cropped_image_2 = equalize_histogram(cropped_image_2)

cropped_image_3 = image[682:1023,0:400,:]
cropped_image_3 = remove_edges(cropped_image_3, 20)
cropped_image_3 = adjust_contrast(cropped_image_3, 1.0, 0)
cropped_image_3 = equalize_histogram(cropped_image_3)

color_image = np.zeros_like(cropped_image_1, dtype=np.uint8)
color_image[:,:,0] = cropped_image_3[:,:,0]  # Kırmızı kanal
color_image[:,:,1] = cropped_image_2[:,:,1]  # Yeşil kanal
color_image[:,:,2] = cropped_image_1[:,:,2]  # Mavi kanal

cv2.imshow("Orijinal Mavi Görüntü", cropped_image_1)
cv2.imshow("Orijinal Yeşil Görüntü", cropped_image_2)
cv2.imshow("Orijinal Kırmızı Görüntü", cropped_image_3)
cv2.imshow("Birleştirilmiş Renklendirilmiş Görüntü", color_image)

cv2.waitKey(0)

end_time = time.time()
total_time = end_time - start_time
print(total_time)
