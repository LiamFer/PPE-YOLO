from ultralytics import YOLO

# Carregar o modelo
model = YOLO("C:\\Users\\Pichau\\Documents\\studies\\YOLO-DETECTOR\\runs\detect\\100epochs\\weights\\last.pt")

# Treinar

model.train(data="config.yaml", epochs=50) 
