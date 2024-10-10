import cv2

image1 = cv2.imread('242.jpg')
image2 = cv2.imread('R.jpg')

image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
print(image1.shape[1], image1.shape[0])

alpha = 0.9  
beta = 1 - alpha  

blended_image = cv2.addWeighted(image1, alpha, image2, beta, 0)

save_image(blended_image, '51.jpg')
