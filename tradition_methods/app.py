import streamlit as st
import numpy as np
from PIL import Image
from utils import (
    retrieve_images_with_absolute_difference,
    retrieve_images_with_euclidean_distance,
    retrieve_images_with_cosine_similarity,
    retrieve_images_with_histogram_feature_similarity,
    cosine_similarity)

#####################################


def show_results(query, method, top_k, size=(224, 224)):
    query = np.array(query.resize(size))

    if method == "Absolute Difference":
        st.markdown("## Results with Absolute Difference")
        results = retrieve_images_with_absolute_difference(query, top_k)
    elif method == "Euclidean Distance":
        st.markdown("## Results with Euclidean Distance")
        results = retrieve_images_with_euclidean_distance(query, top_k)
    elif method == "Cosine Similarity":
        st.markdown("## Results with Cosine Similarity")
        results = retrieve_images_with_cosine_similarity(query, top_k)
        st.write(cosine_similarity(query, query))
    elif method == "Histogram Feature":
        st.markdown("## Results with Histogram Feature")
        results = retrieve_images_with_histogram_feature_similarity(
            query, top_k)

    columns = st.columns(4)

    for i, result in enumerate(results):
        with columns[i % 4]:
            st.image(result[1], caption=f"Top {
                     i+1}: {round(result[0], 4)}", use_column_width=True)


st.title("✨Project-Module2: Image Retrieval with Traditional Methods")

with st.sidebar:
    st.title("Choose Image Retrieval Method")
    method = st.selectbox(
        "Method", ["Absolute Difference", "Euclidean Distance", "Cosine Similarity", "Histogram Feature"])
    top_k = st.number_input("Top K", min_value=1, value=10, step=1)
    image_upload = st.file_uploader("Upload Image")

# Body of the app
if image_upload is not None:
    query = Image.open(image_upload).convert("RGB")
    st.image(query, caption="Uploaded Image")

    show_results(query, method, top_k)
