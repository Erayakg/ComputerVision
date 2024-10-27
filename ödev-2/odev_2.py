import cv2
import numpy as np
import requests
from ultralytics import YOLO

# YOLOv8 modelini başlat
yolo_model = YOLO('yolov8n.pt')

# Görüntüyü indirip analiz eden fonksiyon
def check_image_for_person(image_url):  
    try:
        # Resim URL'sinden görüntü al
        response = requests.get(image_url, stream=True)

        if response.status_code == 200:
            # Resmi belleğe yükle
            img_data = response.content
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                return "Görüntü yüklenemedi"

            # İnsan varlığını tespit etmek için YOLO modelini kullan
            detection_results = yolo_model(img)
            threshold = 0.5  # Güven eşiği

            # İnsan nesnelerini analiz et
            for detection in detection_results:
                detected_boxes = detection.boxes
                for box in detected_boxes:
                    cls_id = int(box.cls[0])  # Sınıf ID'si
                    confidence_score = box.conf[0]  # Güven skoru
                    if yolo_model.names[cls_id] == 'person' and confidence_score > threshold:
                        return f"İNSAN (Güven Skoru: {confidence_score:.2f})"
            return "BAŞKA BİR OBJE"
        else:
            return "Görüntü indirilemedi"
    except Exception as e:
        return f"Bir hata oluştu: {str(e)}"

# İncelenecek görüntü URL listesi
image_urls = [
    "https://imgrosetta.mynet.com.tr/file/19145961/19145961-1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/19205540/19205540-1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/19205569/19205569-1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/19205426/19205426-1200xauto.png",
    "https://imgrosetta.mynet.com.tr/file/19205415/1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/10140538/10140538-1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/14137206/14137206-1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/14125817/1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/19204764/19204764-1200xauto.jpg",
    "https://imgrosetta.mynet.com.tr/file/19204763/1200xauto.jpg"
]

# URL'leri analiz et
for image_url in image_urls:
    analysis_result = check_image_for_person(image_url)
    print(f"{image_url}: {analysis_result}")
