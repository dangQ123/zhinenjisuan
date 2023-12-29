import os
import cv2
import numpy as np

# 类别名和序号的映射
class_mapping = {'metal': 1, 'plastic': 2, 'stone': 3, 'wood': 4}

# 类别名和颜色的映射
color_mapping = {'metal': (0, 0, 255), 'plastic': (0, 255, 0), 'stone': (0, 255, 255), 'wood': (255, 0, 0)}


# 1. 参数估计
def calculate_color_means(train_path):
    class_means = {}

    for class_name, class_idx in class_mapping.items():
        color_means = []

        for image_file in os.listdir(train_path):
            if not image_file.endswith('.png'):
                continue

            image_path = os.path.join(train_path, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 选择特定类别的像素值
            class_pixels = np.where(img == class_idx)

            # 计算颜色均值
            color_mean = np.mean(class_pixels, axis=(0, 1))
            color_means.append(color_mean)

        class_means[class_idx] = np.mean(color_means, axis=0)

    return class_means


# 2. 证据提取
def extract_color_mean(test_image_path):
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        return np.mean(img)
    else:
        raise ValueError(f"Error reading image from {test_image_path}")


# 3. 推理
def classify_texture(test_color_mean, class_means):
    distances = {class_idx: np.linalg.norm(test_color_mean - class_mean)
                 for class_idx, class_mean in class_means.items()}
    return min(distances, key=distances.get)


# 示例用法
data_path = 'output'

# 参数估计
class_means = calculate_color_means(data_path)

# 证据提取
test_color_mean = extract_color_mean(os.path.join(data_path, 'test_image.png'))

# 推理
result_idx = classify_texture(test_color_mean, class_means)

# 获取类别名
result_class = [key for key, value in class_mapping.items() if value == result_idx][0]

print(f'Test image belongs to class: {result_class} (Index: {result_idx})')
print(f'Color associated with the predicted class: {color_mapping[result_class]}')
