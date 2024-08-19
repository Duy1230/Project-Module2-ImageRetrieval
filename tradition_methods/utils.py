import numpy as np
from PIL import Image
import os
import cv2

DATA_PATH = "tradition_methods\\data\\train"
DEFAULT_SIZE = (224, 224)


def read_image(path, size=DEFAULT_SIZE):
    if isinstance(path, str):
        return Image.open(path).convert('RGB').resize(size)
    else:
        raise ValueError("path must be a string")


def absolute_difference(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return np.sum(np.abs(img1 - img2))


def retrieve_images_with_absolute_difference(query, top_k, data_path=DATA_PATH, size=DEFAULT_SIZE):
    images = []
    names = []
    for img in os.listdir(data_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(data_path, img)
            img = read_image(img_path, size)
            images.append(np.array(img))
            names.append(img_path)

    distances = [absolute_difference(query, img) for img in images]
    result = list(zip(distances, names))

    return sorted(result, key=lambda x: x[0])[:top_k]


def euclidean_distance(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return np.sqrt(np.sum((img1 - img2) ** 2))


def retrieve_images_with_euclidean_distance(query, top_k, data_path=DATA_PATH, size=DEFAULT_SIZE):
    images = []
    names = []
    for img in os.listdir(data_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(data_path, img)
            img = read_image(img_path, size)
            images.append(np.array(img))
            names.append(img_path)

    distances = [euclidean_distance(query, img) for img in images]
    result = list(zip(distances, names))

    return sorted(result, key=lambda x: x[0])[:top_k]


def cosine_similarity(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    img1 = img1.flatten().astype(np.float32)
    img2 = img2.flatten().astype(np.float32)
    return np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))


def retrieve_images_with_cosine_similarity(query, top_k, data_path=DATA_PATH, size=DEFAULT_SIZE):
    images = []
    names = []
    for img in os.listdir(data_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(data_path, img)
            img = read_image(img_path, size)
            images.append(np.array(img))
            names.append(img_path)

    distances = [cosine_similarity(query, img) for img in images]
    result = list(zip(distances, names))
    print(distances)

    return sorted(result, key=lambda x: x[0], reverse=True)[:top_k]


def histogram_feature_similarity(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hist_feature_1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist_feature_2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist_feature_1, hist_feature_2, cv2.HISTCMP_CORREL)


def retrieve_images_with_histogram_feature_similarity(query, top_k, data_path=DATA_PATH, size=DEFAULT_SIZE):
    images = []
    names = []
    for img in os.listdir(data_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(data_path, img)
            img = read_image(img_path, size)
            images.append(np.array(img))
            names.append(img_path)

    distances = [histogram_feature_similarity(query, img) for img in images]
    result = list(zip(distances, names))

    return sorted(result, key=lambda x: x[0], reverse=True)[:top_k]
