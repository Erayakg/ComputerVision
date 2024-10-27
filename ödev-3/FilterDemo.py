import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread('Untitled.png', cv2.IMREAD_GRAYSCALE)

# Görüntü başarılı şekilde yüklenmezse hata mesajı ver
if image is None:
    print("Görüntü dosyası bulunamadı veya açılamadı!")
else:
    # Sobel filtresi ile x ve y yönünde türevler al
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitüd hesapla: sqrt(grad_x^2 + grad_y^2)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Magnitüdü normalize et ve görüntü formatına dönüştür
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)

    # Sonucu kaydet veya göster
    cv2.imwrite('magnitüd_resim.jpg', magnitude)
    cv2.imshow('Magnitüd Resim', magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
