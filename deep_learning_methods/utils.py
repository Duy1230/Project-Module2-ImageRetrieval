import os
import numpy as np
from PIL import Image
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

ROOT_DIR = "deep_learning_methods\\data"


def read_image_from_path(path, size=(224, 224)):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)


def get_file_path(dir):
    classes = []
    image_paths = []
    for folder in os.listdir(dir):
        classes.append(folder)
        for file in os.listdir(os.path.join(dir, folder)):
            image_paths.append(os.path.join(dir, folder, file))
    return classes, image_paths


# Initialize the CLIP embedding function once
embedding_function = OpenCLIPEmbeddingFunction(
    model_name="ViT-B-32"
)


def image_embedding(image):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)


def add_embedding(collection, image_paths):
    ids = []
    embeddings = []
    class_names = []

    for idx, image_path in enumerate(image_paths):
        print(f"Embedding {idx+1}/{image_path}")
        image = read_image_from_path(image_path)

        embedding = image_embedding(image.astype(np.uint8))
        # Adjust if using a different OS or path format
        class_name = image_path.split("\\")[-2]

        ids.append(str(idx))  # IDs should be unique and typically strings
        embeddings.append(list(embedding))
        class_names.append(class_name)

    # Add embeddings to the collection
    collection.add(embeddings=embeddings, ids=ids, metadatas=[
                   {"class_name": name} for name in class_names])


def search(collection, image_path, n=5):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    embedding = image_embedding(np.array(image).astype(np.uint8))
    results = collection.query(query_embeddings=[list(embedding)], n_results=n)
    return results


def load_chromadb(collection_name):
    client = PersistentClient(path="deep_learning_methods\\chromadb")
    collection = client.get_collection(name=collection_name)
    return collection


if __name__ == "__main__":
    classes, image_paths = get_file_path(ROOT_DIR)
    chromadb_client = PersistentClient(path="deep_learning_methods\\chromadb")

    collection = chromadb_client.create_collection(
        name="image_collection_cosine",
        metadata={"hnsw:space": "cosine"}
    )

    add_embedding(collection, image_paths)
