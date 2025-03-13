





数据增强相关代码

``` python
import os

import cv2
import numpy as np
import random
import math
from pathlib import Path

# 读取图像并标注框（假设框是以 (x_min, y_min, x_max, y_max) 格式表示）
def load_image_and_labels(image_path, labels):
    image = cv2.imread(image_path)
    return image, labels

# 随机旋转图像及框
def rotate_image_and_boxes(image, boxes, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    new_boxes = []
    for box in boxes:
        classid,x_min, y_min, x_max, y_max = box
        points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        rotated_points = cv2.transform(np.array([points]), M)[0]
        x_min_rot = int(np.min(rotated_points[:, 0]))
        y_min_rot = int(np.min(rotated_points[:, 1]))
        x_max_rot = int(np.max(rotated_points[:, 0]))
        y_max_rot = int(np.max(rotated_points[:, 1]))
        new_boxes.append([classid,x_min_rot, y_min_rot, x_max_rot, y_max_rot])

    return rotated_image, new_boxes

# 随机平移图像及框
def translate_image_and_boxes(image, boxes, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    new_boxes = []
    for box in boxes:
        classid,x_min, y_min, x_max, y_max = box
        new_boxes.append([classid,x_min + tx, y_min + ty, x_max + tx, y_max + ty])

    return translated_image, new_boxes

# 随机缩放图像及框
def scale_image_and_boxes(image, boxes, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    scaled_image = cv2.resize(image, (new_width, new_height))

    new_boxes = []
    for box in boxes:
        classid,x_min, y_min, x_max, y_max = box
        x_min_scaled = int(x_min * scale_factor)
        y_min_scaled = int(y_min * scale_factor)
        x_max_scaled = int(x_max * scale_factor)
        y_max_scaled = int(y_max * scale_factor)
        new_boxes.append([classid,x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])

    return scaled_image, new_boxes

# 随机水平翻转图像及框
def flip_image_and_boxes(image, boxes):
    flipped_image = cv2.flip(image, 1)
    flipped_boxes = []
    for box in boxes:
        classid,x_min, y_min, x_max, y_max = box
        flipped_boxes.append([classid,image.shape[1] - x_max, y_min, image.shape[1] - x_min, y_max])
    return flipped_image, flipped_boxes



# 模拟标注框
def generate_labels(image):
    height, width = image.shape[:2]
    # 假设框的位置
    boxes = [[int(width * 0.3), int(height * 0.3), int(width * 0.7), int(height * 0.7)]]  # 一个中间的框
    return boxes

# YOLO 格式标签转换函数
def convert_to_yolo_format(boxes, image_width, image_height):
    yolo_labels = []
    for box in boxes:
        _,x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # 将坐标归一化
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        yolo_labels.append([0, x_center, y_center, width, height])  # 假设 class_id 为 0
    return yolo_labels







import os
import cv2
import numpy as np
import random
from pathlib import Path

# 读取图像并标注框（假设框是以 (x_min, y_min, x_max, y_max) 格式表示，且包括类别）
def load_image_and_labels(image_path, label_path):
    image = cv2.imread(image_path)
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            # YOLO格式：class_id, x_center, y_center, width, height
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 将中心坐标和宽高转换为 (x_min, y_min, x_max, y_max)
                image_height, image_width = image.shape[:2]
                x_min = int((x_center - width / 2) * image_width)
                y_min = int((y_center - height / 2) * image_height)
                x_max = int((x_center + width / 2) * image_width)
                y_max = int((y_center + height / 2) * image_height)

                boxes.append([class_id, x_min, y_min, x_max, y_max])

    return image, boxes

# 绘制框，使用不同颜色区分类别
def draw_boxes(image, boxes):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # 不同类别使用不同颜色
    for box in boxes:
        class_id, x_min, y_min, x_max, y_max = box
        color = colors[class_id % len(colors)]  # 确保颜色循环
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image

# YOLO 格式标签转换函数
def convert_to_yolo_format(boxes, image_width, image_height):
    yolo_labels = []
    for box in boxes:
        class_id, x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # 将坐标归一化
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        yolo_labels.append([class_id, x_center, y_center, width, height])  # 使用原始类别
    return yolo_labels

# 数据增强函数（旋转、平移、缩放、翻转等）
def augment_and_save(image_path, label_path, output_path):
    # 获取原图像文件名的前缀（不带扩展名）
    image_name = Path(image_path).stem  # 获取图像的文件名不带扩展名
    image_extension = Path(image_path).suffix  # 获取图像文件的扩展名（例如 .jpg）

    # 读取图像和标签
    image, boxes = load_image_and_labels(image_path, label_path)
    image_height, image_width = image.shape[:2]

    # 1. 旋转
    rotated_image, rotated_boxes = rotate_image_and_boxes(image, boxes, angle=random.randint(-90, 90))
    # rotated_image = draw_boxes(rotated_image, rotated_boxes)
    yolo_labels = convert_to_yolo_format(rotated_boxes, image_width, image_height)
    cv2.imwrite(output_path / f'{image_name}_rotate{image_extension}', rotated_image)
    save_yolo_labels(output_path / f'{image_name}_rotate', yolo_labels)

    # # 2. 平移
    # translated_image, translated_boxes = translate_image_and_boxes(image, boxes, tx=random.randint(-50, 50), ty=random.randint(-50, 50))
    # # translated_image = draw_boxes(translated_image, translated_boxes)
    # yolo_labels = convert_to_yolo_format(translated_boxes, image_width, image_height)
    # cv2.imwrite(output_path / f'{image_name}_translate{image_extension}', translated_image)
    # save_yolo_labels(output_path / f'{image_name}_translate', yolo_labels)

    # 3. 缩放
    scale_factor = random.uniform(0.8, 1.2)  # 随机缩放
    scaled_image, scaled_boxes = scale_image_and_boxes(image, boxes, scale_factor)
    # scaled_image = draw_boxes(scaled_image, scaled_boxes)
    yolo_labels = convert_to_yolo_format(scaled_boxes, image_width, image_height)
    cv2.imwrite(output_path / f'{image_name}_scale{image_extension}', scaled_image)
    save_yolo_labels(output_path / f'{image_name}_scale', yolo_labels)

    # # 4. 水平翻转
    # flipped_image, flipped_boxes = flip_image_and_boxes(image, boxes)
    # # flipped_image = draw_boxes(flipped_image, flipped_boxes)
    # yolo_labels = convert_to_yolo_format(flipped_boxes, image_width, image_height)
    # cv2.imwrite(output_path / f'{image_name}_flip{image_extension}', flipped_image)
    # save_yolo_labels(output_path / f'{image_name}_flip', yolo_labels)

# 保存YOLO格式的标签
def save_yolo_labels(output_path, yolo_labels):
    label_file = output_path.with_suffix('.txt')
    with open(label_file, 'w') as f:
        for label in yolo_labels:
            f.write(" ".join(map(str, label)) + "\n")

# 批量数据增强
def aug_batch(img_dir, label_dir, output_path, num_aug=10):
    for file in os.listdir(img_dir):
        # 读取并增强图片的路径
        image_path = os.path.join(img_dir, file)
        # 判断是否是jpg 结尾
        if  file.endswith('.jpg'):
            label_path = os.path.join(label_dir, file.replace('.jpg', '.txt'))
        elif file.endswith('.png'):
            label_path = os.path.join(label_dir, file.replace('.png', '.txt'))
        # 输出路径
        # output_path = Path('D:/python-project/datasets/custom-data/yolo-qr-v6/augment')
        output_path.mkdir(parents=True, exist_ok=True)

        # 执行数据增强
        augment_and_save(image_path, label_path, output_path)
        print(f"Processed: {file}")

if __name__ == '__main__':
    img_dir = r'D:\python-project\datasets\custom-number-data\labeld\images'  # 原始图片文件夹路径
    label_dir = r'D:\python-project\datasets\custom-number-data\labeld\labels'  # 原始标签文件夹路径
    output_dir =Path( r'D:\python-project\datasets\custom-number-data\labeld\aug' ) # 输出文件夹路径
    aug_batch(img_dir=img_dir, label_dir=label_dir,output_path=output_dir)

```

