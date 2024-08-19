# Image Retrieval Project

This project implements two main approaches for image retrieval:

1. **Traditional Methods**: 
   - L1 and L2 Distance: Measures the absolute and squared differences between pixel values.
   - Cosine Similarity: Measures the cosine of the angle between two vectors.
   - Histogram Comparison: Compares the distribution of color intensities in images.

2. **Deep Learning Method**: 
   - Utilizes CLIP (Contrastive Language-Image Pretraining) to embed images.
   - The embedded feature vectors are stored in ChromaDB for efficient retrieval.

## Prerequisites

Ensure you have Python installed (preferably Python 3.12 or above).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Duy1230/Project-Module2-ImageRetrieval.git
   cd Project-Module2-ImageRetrieval
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit application:
    ```bash
    streamlit run deep_learning_methods\\app.py
    streamlit run tradition_methods\\app.py
    ```