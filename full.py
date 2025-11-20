import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import st_folium

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Cloud Cluster Prediction System",
    page_icon="☁",
    layout="wide"
)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("☁ Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Upload & Predict", "Risk Map", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤ using Streamlit")


# GLOBAL VARIABLES
df_global = None
X_img_global = None
pred_global = None
img_names_global = None


# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":
    st.title("☁ AI/ML Cloud Cluster Prediction System")
    st.markdown(
        """
        ### Welcome to the Cloudburst Detection Dashboard  
        This system uses:
        - 🛰 Satellite Images  
        - 📄 Meteorological Excel Data  
        - 🤖 Deep Learning (CNN + Tabular model)  
        - 🌧 Risk Prediction & Geo-Mapping  

        Navigate using the sidebar to upload data and run predictions.

      Tropical clouds are dense and towering formations that occur in regions near the equator. They play a crucial role in Earth’s energy balance and rainfall patterns. Monitoring these clouds is vital for understanding
      tropical weather systems, predicting storms, and studying climate change.

      The Tropical Cloud Detection System aims to provide a simple yet effective platform to visualize and
      understand tropical weather phenomena. By combining half-hourly satellite data with advanced machine
      learning algorithms (in future versions), this project aspires to predict cloud cluster development,
      assess rainfall potential, and enhance disaster preparedness.

        """
    )

    st.image(
        "https://sc0.blr1.cdn.digitaloceanspaces.com/article/129896-zhpiafcrds-1572257955.jpeg",
        use_container_width=True,
        caption="Real Satellite Cloud Image"
    )

    st.markdown("---")
    st.markdown("### Start by selecting *Upload & Predict* from the sidebar.")


# =====================================================
# UPLOAD + TRAIN + PREDICT PAGE
# =====================================================
elif page == "Upload & Predict":

    st.title("📤 Upload Files & Predict Cloudburst Risk")

    excel_file = st.file_uploader("📄 Upload Meteorological Excel (.xlsx)", type=["xlsx"])
    uploaded_images = st.file_uploader("🖼 Upload Satellite Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if excel_file and uploaded_images:

        # Load Excel
        df = pd.read_excel(excel_file)
        st.success(f"Loaded {len(df)} meteorological data rows.")

        required_cols = {"Latitude", "Longitude", "2m temperature", "Total precipitation", "Label"}
        if not required_cols.issubset(df.columns):
            st.error("Excel columns missing! Required: " + ", ".join(required_cols))
            st.stop()

        # Process tabular data
        X_tab = df[["Latitude", "Longitude", "2m temperature", "Total precipitation"]]
        y = df["Label"]

        scaler = StandardScaler()
        X_tab_scaled = scaler.fit_transform(X_tab)

        # Load images
        imgs, names = [], []
        for file in uploaded_images:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))
            img = img.astype("float32") / 255.0
            imgs.append(img)
            names.append(file.name)

        X_img = np.array(imgs)
        X_tab_scaled = X_tab_scaled[:len(X_img)]
        y = y[:len(X_img)]

        st.info(f"🧠 Loaded {len(X_img)} image-tabular pairs.")

        # -------------------------
        # HYBRID MODEL
        # -------------------------
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

        st.subheader("🔧 Training the Model (Demo Training - 5 epochs)")
        with st.spinner("Training…"):
            model.fit([X_img, X_tab_scaled], y, epochs=5, batch_size=4, verbose=0)

        st.success("🎉 Model Trained Successfully!")

        # Predict
        predictions = model.predict([X_img, X_tab_scaled]).flatten()
        df["Predicted_Risk"] = predictions

        st.subheader("🖼 Prediction Results")
        cols = st.columns(3)

        for i, img in enumerate(X_img):
            risk = predictions[i]
            color = (0, 0, 255) if risk > 0.5 else (0, 255, 0)

            display_img = (img * 255).astype(np.uint8)
            overlay = display_img.copy()

            cv2.rectangle(overlay, (0, 0), (128, 25), color, -1)
            cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
            cv2.putText(display_img, f"Risk: {risk:.2f}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            with cols[i % 3]:
                st.image(display_img, caption=names[i])

        # SAVE GLOBAL VARIABLES for MAP PAGE
        st.session_state["df"] = df
        st.session_state["predictions_loaded"] = True
        st.success("Prediction Saved! Go to *Risk Map* Page.")


# =====================================================
# MAP PAGE
# =====================================================
elif page == "Risk Map":

    st.title("🗺 Cloudburst Risk Map")

    if "predictions_loaded" not in st.session_state:
        st.warning("⚠ Run prediction first! Go to 'Upload & Predict' page.")
        st.stop()

    df = st.session_state["df"]

    m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=6)

    for _, row in df.iterrows():
        col = "red" if row["Predicted_Risk"] > 0.5 else "green"
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=7,
            color=col,
            fill=True,
            fill_opacity=0.8,
            popup=f"Risk Score: {row['Predicted_Risk']:.2f}"
        ).add_to(m)

    st_folium(m, width=750, height=500)


# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":
    st.title("ℹ About This System")

    st.markdown(
        """
        ### 🌧 Cloudburst Prediction System  
        This platform combines:
        - CNN-based satellite image analysis  
        - Meteorological data modeling  
        - Hybrid Deep Learning  
        - Interactive Geo-Mapping  

        The Tropical Cloud Detection System using Satellite Half-hourly Data is a research-oriented web project designed
      to detect, analyze, and visualize tropical cloud formations using frequent satellite observations. The project aims
      to support weather research and climate analysis by providing clear visual data of cloud behavior across tropical regions.

      The system focuses on processing half-hourly satellite imagery — this allows the detection of fast-changing
      cloud clusters and atmospheric patterns. By visualizing these transitions, meteorologists can monitor the
      evolution of storms, cloud thickness, and possible cyclone formations in real-time.


      In its next version, this project will integrate Machine Learning and Deep Learning models to automatically
      classify cloud types and forecast severe weather events. Frameworks such as TensorFlow and PyTorch will be utilized to
      train CNN models on satellite datasets for automated tropical cloud prediction.



        Built for research, disaster prediction, and real-time monitoring.

        """
    )

    st.markdown("---")
    st.markdown("*Version:* 2.0")  
    st.markdown("*Team members:Rishika sharma\nVedashree R\n,Shiva kumar\n,Yudhil krishna ")