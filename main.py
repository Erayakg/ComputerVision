from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = "C:/Users/erayb/PycharmProjects/pythonProject2/1200px-LT_471_(LTZ_1471)_Arriva_London_New_Routemaster_(19522859218).jpg"

# Run inference on the source
results = model(source)  # list of Results objects