import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tempfile
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Cloud Cluster Prediction", layout="wide")
st.title("☁ AI/ML Cloud Cluster Prediction System")
st.write("Upload satellite images and meteorological data to detect and predict cloudburst risk zones.")

# --------------------------
# File Upload Section
# --------------------------
excel_file = st.file_uploader("📄 Upload Meteorological Excel File (.xlsx)", type=["xlsx"])
uploaded_images = st.file_uploader("🖼 Upload Satellite Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if excel_file and uploaded_images:
    # Read Excel
    df = pd.read_excel(excel_file)
    st.success(f"Loaded {len(df)} meteorological records.")
    
    # Check columns
    required_cols = {"Latitude", "Longitude", "2m temperature", "Total precipitation", "Label"}
    if not required_cols.issubset(df.columns):
        st.error(f"Excel file must contain columns: {', '.join(required_cols)}")
        st.stop()

    # Preprocess tabular data
    X_tab = df[["Latitude", "Longitude", "2m temperature", "Total precipitation"]]
    y = df["Label"]
    scaler = StandardScaler()
    X_tab_scaled = scaler.fit_transform(X_tab)

    # Load Images
    images, valid_names = [], []
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_images:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        images.append(img)
        valid_names.append(uploaded_file.name)

    X_img = np.array(images)
    X_tab_scaled = X_tab_scaled[:len(X_img)]
    y = y[:len(X_img)]
    st.info(f"✅ Loaded {len(X_img)} image–data pairs successfully.")

    # --------------------------
    # Define Simple Hybrid Model
    # --------------------------
    img_input = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(32, (3, 3), activation="relu")(img_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)

    tab_input = layers.Input(shape=(X_tab_scaled.shape[1],))
    y_tab = layers.Dense(32, activation="relu")(tab_input)
    y_tab = layers.Dense(16, activation="relu")(y_tab)

    combined = layers.concatenate([x, y_tab])
    z = layers.Dense(32, activation="relu")(combined)
    output = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model(inputs=[img_input, tab_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    st.subheader("⚙ Training the Model (demo training, small dataset)...")
    with st.spinner("Training in progress..."):
        model.fit([X_img, X_tab_scaled], y, epochs=5, batch_size=4, verbose=0)
    st.success("✅ Model trained successfully!")

    # --------------------------
    # Predict
    # --------------------------
    predictions = model.predict([X_img, X_tab_scaled]).flatten()
    df["Predicted_Risk"] = predictions

    # --------------------------
    # Visualize Images
    # --------------------------
    st.subheader("🖼 Visualized Predictions")
    cols = st.columns(3)
    for i, img in enumerate(X_img):
        color = (0, 255, 0) if predictions[i] < 0.5 else (0, 0, 255)
        label_text = f"Risk: {predictions[i]:.2f}"
        img_disp = (img * 255).astype(np.uint8)
        overlay = img_disp.copy()
        cv2.rectangle(overlay, (0, 0), (128, 25), color, -1)
        cv2.addWeighted(overlay, 0.4, img_disp, 0.6, 0, img_disp)
        cv2.putText(img_disp, label_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        with cols[i % 3]:
            st.image(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB), caption=valid_names[i])

    # --------------------------
    # Visualize on Map
    # --------------------------
    st.subheader("🗺 Predicted Cloudburst Risk Map")

    m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=6)
    for _, row in df.iterrows():
        color = "red" if row["Predicted_Risk"] > 0.5 else "green"
        label = f"Risk: {row['Predicted_Risk']:.2f}"
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=label
        ).add_to(m)

    st_folium(m, width=700, height=500)