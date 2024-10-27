from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = "C:/Users/erayb/PycharmProjects/pythonProject2/hafta-1/1200px-LT_471_(LTZ_1471)_Arriva_London_New_Routemaster_(19522859218).jpg"

# Run inference on the source6+9cont
results = model(source)  # list of Results objects

results = model(["C:/Users/erayb/PycharmProjects/pythonProject2/hafta-1/1200px-LT_471_(LTZ_1471)_Arriva_London_New_Routemaster_(19522859218).jpg"])  # return a list of Results objects


for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk