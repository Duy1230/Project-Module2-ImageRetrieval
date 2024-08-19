import streamlit as st
import numpy as np
from PIL import Image
from utils import (
    get_file_path,
    search,
    load_chromadb)

#####################################

collection_l2 = load_chromadb('image_collection')
collection_cosine = load_chromadb('image_collection_cosine')
classes, image_paths = get_file_path("deep_learning_methods\\data")


def show_results(query, method, top_k):
    if method == "L2-Distance":
        st.markdown("## Results with Euclidean Distance")
        results = search(collection_l2, query, top_k)
    elif method == "Cosine Similarity":
        st.markdown("## Results with Cosine Similarity")
        results = search(collection_cosine, query, top_k)

    ids = results['ids'][0]
    distances = results['distances'][0]
    classes = results['metadatas'][0]

    columns = st.columns(4)

    for i in range(len(ids)):
        with columns[i % 4]:
            st.image(image_paths[int(ids[i])], caption=f"Top {
                     i+1}: {round(distances[i], 4)} - {classes[i]['class_name']}", use_column_width=True)


st.title("✨Project-Module2: Image Retrieval with Traditional Methods")

with st.sidebar:
    st.title("Choose Image Retrieval Method")
    method = st.selectbox(
        "Method", ["L2-Distance",  "Cosine Similarity"])
    top_k = st.number_input("Top K", min_value=1, value=10, step=1)
    image_upload = st.file_uploader("Upload Image")

# Body of the app


if image_upload is not None:
    display_image = Image.open(image_upload)
    st.image(display_image, caption="Uploaded Image")

    show_results(image_upload, method, top_k)
