import streamlit as st
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.spatial import distance
import json

# Load color data from JSON file
with open('Colors.json') as f:
    color_data = json.load(f)

def save_uploaded_file(uploaded_file):
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

def get_dominant_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply thresholding to the grayscale image
    threshold_value = 100
    image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]

    # Flatten the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Apply KMeans clustering to get dominant colors
    n_colors = 1  # Predict only the most dominant color
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    # Get the index of the largest cluster
    largest_cluster_index = np.argmax(np.bincount(kmeans.labels_))

    # Get the dominant color as the centroid of the largest cluster
    dominant_color = list(map(int, kmeans.cluster_centers_[largest_cluster_index]))

    return dominant_color

# Function to find the closest color in the dataset based on RGB values
def find_closest_color(rgb_values, color_data):
    min_distance = float('inf')
    closest_color = None

    for color in color_data["data"]:
        color_code = color["color_code"]
        color_code_rgb = np.array([int(color_code[i:i + 2], 16) for i in (0, 2, 4)])
        current_distance = distance.euclidean(rgb_values, color_code_rgb)

        if current_distance < min_distance:
            min_distance = current_distance
            closest_color = color

    return closest_color

def main():
    st.title("Color Detection App")
    uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

    # ...

    if uploaded_image is not None:
        save_uploaded_file(uploaded_image)
        image_path = os.path.join("tempDir", uploaded_image.name)

        dominant_color = get_dominant_color(image_path)
        detected_color = None

        # Find the closest color in the dataset based on RGB values
        closest_color = find_closest_color(dominant_color, color_data)

        st.write("Predicted Color:")
        st.write(f"RGB Values: {dominant_color}")

        if closest_color is not None:
            st.write("\nColor Information:")
            st.write(f"Color Name: {closest_color['color_name']}")
            st.write(f"Color ID: {closest_color['color_id']}")
            st.write(f"Color Code: {closest_color['color_code']}")
            st.write(f"Color Type: {closest_color['color_type']}")
        else:
            st.write("Color information not found for the detected color.")

if __name__ == "__main__":
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")
    main()
