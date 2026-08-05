import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# --------------------------------------------------
# Custom website styling
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

h1 {
    text-align: center;
    font-size: 42px;
}

h2, h3 {
    margin-top: 20px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

[data-testid="stMetric"] {
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Iris dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# Create and train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# --------------------------------------------------
# Website title
# --------------------------------------------------

st.title("🌸 Iris Flower Classification")

st.write(
    "### Machine Learning Prediction App"
)

st.write(
    "Enter the four measurements of an Iris flower "
    "and our machine learning model will predict its species."
)

st.divider()

# --------------------------------------------------
# Input section
# --------------------------------------------------

st.subheader("📋 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.2,
        step=0.1
    )

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Flower", use_container_width=True):

    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

    species = iris.target_names[prediction]

    confidence = probabilities[prediction] * 100

    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.divider()

    st.subheader("🔮 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.success(
            f"🌸 Predicted Species: {species.capitalize()}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    with result_col2:

        image_path = f"images/{species}.jpg"

        st.image(
            image_path,
            caption=f"Iris {species.capitalize()}",
            use_container_width=True
        )

    # --------------------------------------------------
    # Prediction Probabilities
    # --------------------------------------------------

    st.subheader("📊 Prediction Probabilities")

    for i, name in enumerate(iris.target_names):

        probability = probabilities[i] * 100

        st.write(
            f"**{name.capitalize()}**: {probability:.2f}%"
        )

        st.progress(float(probabilities[i]))

# --------------------------------------------------
# Information section
# --------------------------------------------------

st.divider()

st.subheader("🌺 About the Iris Species")

species_col1, species_col2, species_col3 = st.columns(3)

with species_col1:
    st.markdown(
        """
        ### 🌸 Setosa

        - Small petals
        - Short petal length
        - One of the three Iris species
        """
    )

with species_col2:
    st.markdown(
        """
        ### 🌼 Versicolor

        - Medium-sized petals
        - Moderate petal length
        - One of the three Iris species
        """
    )

with species_col3:
    st.markdown(
        """
        ### 🌺 Virginica

        - Larger petals
        - Longer petal length
        - One of the three Iris species
        """
    )


# --------------------------------------------------
# How the Project Works
# --------------------------------------------------

st.divider()

st.subheader("🧠 How This Project Works")

st.write(
    """
    **1️⃣ Enter Flower Measurements**

    The user enters four measurements:
    Sepal Length, Sepal Width, Petal Length, and Petal Width.

    **2️⃣ Machine Learning Model**

    A K-Nearest Neighbors (KNN) model processes the measurements.

    **3️⃣ Compare with Training Data**

    KNN compares the new flower with nearby examples from
    the Iris dataset.

    **4️⃣ Predict the Species**

    The model predicts one of three species:
    Setosa, Versicolor, or Virginica.

    **5️⃣ Display the Result**

    The website displays the predicted species,
    confidence, probabilities, and matching flower image.
    """
)

# --------------------------------------------------
# Project information
# --------------------------------------------------

st.divider()

st.subheader("🤖 About This Project")

st.write(
    """
    This project uses the Iris dataset and a
    K-Nearest Neighbors (KNN) machine learning model
    to classify Iris flowers into three species:

    • Setosa  
    • Versicolor  
    • Virginica
    """
)

st.info(
    "Built using Python, Scikit-learn and Streamlit."
)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "🌸 Iris Flower Classification | Machine Learning Project"
)