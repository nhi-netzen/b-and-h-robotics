from IPython import get_ipython
from IPython.display import display
import shlex
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import json
import cv2
import numpy as np
import glob
import shutil
import zipfile
from PIL import Image

# Upload file
filename = "/data.zip"
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('/extracted_files')

# # Mount Google Drive
# drive.mount('/drive')

# Huấn luyện mô hình YOLO
model = YOLO("yolov8n.pt")  # hoặc yolov8s.pt
model.train(data="/extracted_files/data.yaml", epochs=50, imgsz=640)

# Dự đoán ảnh test bằng YOLO
model_yolo = YOLO("/runs/detect/train/weights/best.pt")
results = model_yolo.predict("/extracted_files/test/images/", conf=0.5)

# Hàm cắt ảnh dựa trên kết quả YOLO
def crop_images(image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for img_path in image_paths:
        img = cv2.imread(img_path)
        results = model_yolo.predict(img)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]

            class_id = classes[i]
            class_name = model_yolo.names[class_id]

            class_folder = os.path.join(save_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)

            base_name = os.path.basename(img_path).split('.')[0]
            save_path = os.path.join(class_folder, f"{base_name}_{i}.jpg")
            cv2.imwrite(save_path, crop_img)

# Crop ảnh train và valid
crop_images("/extracted_files/train/images", "/data_cropped/train")
crop_images("/extracted_files/valid/images", "/data_cropped/val")

# Kiểm tra nhãn bị thiếu trong thư mục valid
train_dir = "/data_cropped/train"
val_dir = "/data_cropped/val"
train_labels = sorted(os.listdir(train_dir))
val_labels = sorted(os.listdir(val_dir))
missing_labels = set(train_labels) - set(val_labels)
print("✅ Nhãn bị thiếu trong valid:", missing_labels)

# Tạo thư mục cho các nhãn bị thiếu trong valid
for label in missing_labels:
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)

# Copy một số ảnh từ train sang valid cho các nhãn bị thiếu
for label in missing_labels:
    src_folder = os.path.join(train_dir, label)
    dst_folder = os.path.join(val_dir, label)
    images = glob.glob(f"{src_folder}/*.jpg")[:10]  # Lấy 10 ảnh đầu tiên
    for img_path in images:
        shutil.copy(img_path, dst_folder)

# In số lượng ảnh mỗi nhãn trong thư mục train (sau khi có thể đã copy)
print("\nSố lượng ảnh mỗi nhãn trong train:")
for label in sorted(os.listdir(train_dir)):
    count = len(os.listdir(os.path.join(train_dir, label)))
    print(f"{label}: {count} ảnh")


# Cấu hình ImageDataGenerator cho augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)


# Tạo bộ dữ liệu cho CNN
train_dir_cnn = '/data_cropped/train' # Sử dụng thư mục train sau khi xử lý
val_dir_cnn = '/data_cropped/val' # Sử dụng thư mục valid sau khi xử lý

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir_cnn,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir_cnn,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Mô hình CNN
model_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Thêm Dropout như trong đoạn code cuối
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile mô hình CNN
model_cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Huấn luyện mô hình CNN
history = model_cnn.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20 # Sử dụng 20 epoch như trong lần huấn luyện thứ hai
)

# Lưu model CNN
model_cnn.save("/food_cnn_model.h5")


# Dự đoán và tính tổng tiền dựa trên menu
# Load menu giá món ăn
menu_list=[
    {"label":"ca_kho","price":15000},
    {"label":"canh_bap_cai","price":5000},
    {"label":"canh_bau","price":5000},
    {"label":"canh_bi_do","price":5000},
    {"label":"canh_cai","price":5000},
    {"label":"canh_chua","price":5000},
    {"label":"com_trang","price":5000},
    {"label":"dau_hu_sot_ca","price":7000},
    {"label":"dua_leo","price":5000},
    {"label":"ga_chien","price":25000},
    {"label":"lap_xuong","price":15000},
    {"label":"mang_xao","price":10000},
    {"label":"muop_xao","price":10000},
    {"label":"rau_muong_xao","price":15000},
    {"label":"thit_kho","price":20000},
    {"label":"thit_kho_trung","price":30000},
    {"label":"tom_kho","price":30000},
    {"label":"trung_chien","price":10000}
]
menu = {item["label"]: item["price"] for item in menu_list}

# Load ảnh từ thư mục test
image_paths = glob.glob('/extracted_files/test/images/*.jpg')

all_results = []

for img_path in image_paths:
    # Đọc ảnh
    img = cv2.imread(img_path)

    # YOLO detect
    results = model_yolo.predict(img)[0]  # Chỉ lấy kết quả đầu
    labels = [model_yolo.model.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
    total = sum(menu.get(label.strip(), 0) for label in labels)

    print(f"\nẢnh: {img_path}")
    print("Món ăn nhận diện:", labels)
    print("Tổng tiền:", total, "VNĐ")

    # In nhãn nhận diện từ YOLO cho từng box
    for box in results.boxes:
        class_id = int(box.cls.cpu().numpy()[0])
        label = model_yolo.model.names[class_id]
        print("YOLO nhận diện:", label)

# Code để tải folder content xuống (có thể chạy riêng sau khi hoàn thành các bước trên)
# from google.colab import files
# !zip -r /content.zip /
# files.download('/content.zip')

# Code để upload file (đã được đưa lên đầu)
# from google.colab import files
# files.upload()