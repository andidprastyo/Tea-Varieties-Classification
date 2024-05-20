import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('./TEH1.csv')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Remove extra spaces
    return df

df = load_data()

# Encode the 'TEH' column (tea types)
label_encoder = LabelEncoder()
df['TEH'] = label_encoder.fit_transform(df['TEH'])

# Splitting the data
X = df.drop('TEH', axis=1)
y = df['TEH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit App
st.title("Tea Variant Classification")

# Display dataset and visualization
expander = st.expander("Dataset and Visualizations")
with expander:
    st.write("### Dataset")
    st.write(df.head())

    st.write("### Gas Sensor Readings Distribution")
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    sns.histplot(df.columns[1], bins=30, kde=True, ax=axs[0, 0]).set_title('MQ3 Distribution')
    sns.histplot(df.columns[2], bins=30, kde=True, ax=axs[0, 1]).set_title('MQ4 Distribution')
    sns.histplot(df.columns[3], bins=30, kde=True, ax=axs[1, 0]).set_title('MQ5 Distribution')
    sns.histplot(df.columns[4], bins=30, kde=True, ax=axs[1, 1]).set_title('MQ135 Distribution')
    st.pyplot(fig)

    st.write("### Pair Plot of Gas Sensor Readings")
    fig_pairplot = sns.pairplot(df, hue='TEH', diag_kind='kde')
    st.pyplot(fig_pairplot.fig)

# User input for prediction
st.write("### Predict Tea Variant")
mq3 = st.number_input('Enter MQ3 Reading', min_value=0)
mq4 = st.number_input('Enter MQ4 Reading', min_value=0)
mq5 = st.number_input('Enter MQ5 Reading', min_value=0)
mq135 = st.number_input('Enter MQ135 Reading', min_value=0)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[mq3, mq4, mq5, mq135]], columns=[1, 2, 3, 4])
    prediction = clf.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    st.write(f"The predicted tea variant is: **{prediction_label}**")

# Model evaluation metrics
expander_metrics = st.expander("Model Evaluation Metrics")
with expander_metrics:
    y_test_pred = clf.predict(X_test)

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    st.write("### Confusion Matrix")
    matrix = confusion_matrix(y_test, y_test_pred)
    fig_conf_matrix, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, fmt='d', ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(fig_conf_matrix)
