import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import ast

# Load and preprocess data
@st.cache_data
def load_data():
    disease_features = pd.read_csv('disease_features.csv')
    one_hot_encoded = pd.read_csv('encoded_output2.csv')
    
    def parse_list(text):
        if pd.isna(text) or text.strip() == '[]':
            return []
        try:
            return ast.literal_eval(text.strip())
        except:
            return []

    disease_features['Risk Factors List'] = disease_features['Risk Factors'].apply(parse_list)
    disease_features['Symptoms List'] = disease_features['Symptoms'].apply(parse_list)
    disease_features['Signs List'] = disease_features['Signs'].apply(parse_list)
    
    disease_features['Risk Factors Text'] = disease_features['Risk Factors List'].apply(lambda x: ' '.join(x))
    disease_features['Symptoms Text'] = disease_features['Symptoms List'].apply(lambda x: ' '.join(x))
    disease_features['Signs Text'] = disease_features['Signs List'].apply(lambda x: ' '.join(x))
    disease_features.fillna('', inplace=True)
    
    return disease_features, one_hot_encoded

disease_features, one_hot_encoded = load_data()

@st.cache_data
def get_vectorizers_and_features():
    tfidf_risk = TfidfVectorizer(min_df=1)
    tfidf_symptoms = TfidfVectorizer(min_df=1)
    tfidf_signs = TfidfVectorizer(min_df=1)
    
    risk_matrix = tfidf_risk.fit_transform(disease_features['Risk Factors Text'])
    symptoms_matrix = tfidf_symptoms.fit_transform(disease_features['Symptoms Text'])
    signs_matrix = tfidf_signs.fit_transform(disease_features['Signs Text'])
    
    risk_df = pd.DataFrame(risk_matrix.toarray(), columns=[f"Risk_{f}" for f in tfidf_risk.get_feature_names_out()])
    symptoms_df = pd.DataFrame(symptoms_matrix.toarray(), columns=[f"Symptom_{f}" for f in tfidf_symptoms.get_feature_names_out()])
    signs_df = pd.DataFrame(signs_matrix.toarray(), columns=[f"Sign_{f}" for f in tfidf_signs.get_feature_names_out()])
    
    tfidf_combined = pd.concat([disease_features[['Disease']], risk_df, symptoms_df, signs_df], axis=1)
    return tfidf_risk, tfidf_symptoms, tfidf_signs, tfidf_combined

risk_vectorizer, symptoms_vectorizer, signs_vectorizer, tfidf_combined = get_vectorizers_and_features()


X_tfidf = tfidf_combined.iloc[:, 1:].values
X_onehot = one_hot_encoded.iloc[:, 1:].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(disease_features['Disease'])

# Train KNN models
@st.cache_resource
def train_knn(encoding_type, n_neighbors=5, metric='euclidean'):
    if encoding_type == 'TF-IDF':
        X = X_tfidf
    else:
        X = X_onehot
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X, y)
    return knn

st.title("Disease Classification with KNN")
st.write("""
This application uses K-Nearest Neighbors (KNN) to classify diseases based on their:
- Risk factors
- Symptoms
- Clinical signs
""")

st.sidebar.header("Model Configuration")
encoding_type = st.sidebar.selectbox("Feature Encoding", ['TF-IDF', 'One-hot'])
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
metric = st.sidebar.selectbox("Distance Metric", ['euclidean', 'manhattan', 'cosine'])

model = train_knn(encoding_type, n_neighbors, metric)

st.header("Enter Patient Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Risk Factors")
    risk_factors = st.text_area("Enter risk factors (comma separated)", 
                               "smoking, obesity, family history")

with col2:
    st.subheader("Symptoms")
    symptoms = st.text_area("Enter symptoms (comma separated)", 
                           "chest pain, shortness of breath, fatigue")

with col3:
    st.subheader("Clinical Signs")
    signs = st.text_area("Enter clinical signs (comma separated)", 
                        "elevated blood pressure, rapid heart rate")

if st.button("Predict Disease"):
    if encoding_type == 'TF-IDF':
        risk_input = ' '.join([x.strip() for x in risk_factors.split(',')])
        symptoms_input = ' '.join([x.strip() for x in symptoms.split(',')])
        signs_input = ' '.join([x.strip() for x in signs.split(',')])
        
        risk_vec = risk_vectorizer.transform([risk_input])
        symptoms_vec = symptoms_vectorizer.transform([symptoms_input])
        signs_vec = signs_vectorizer.transform([signs_input])
        
        input_features = np.hstack([risk_vec.toarray(), symptoms_vec.toarray(), signs_vec.toarray()])
    else:
        input_features = np.zeros(X_onehot.shape[1])
        
        all_features = one_hot_encoded.columns[1:]  
        
        user_inputs = ([x.strip().lower() for x in risk_factors.split(',')] + 
                      [x.strip().lower() for x in symptoms.split(',')] + 
                      [x.strip().lower() for x in signs.split(',')])
        
        for i, feature in enumerate(all_features):
            feature_lower = feature.lower()
            if any(user_input in feature_lower for user_input in user_inputs):
                input_features[i] = 1
    
    prediction = model.predict(input_features.reshape(1, -1))
    disease = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"Predicted Disease: **{disease}**")
    
    distances, indices = model.kneighbors(input_features.reshape(1, -1))
    st.subheader("Similar Diseases:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        disease_name = label_encoder.inverse_transform([idx])[0]
        if i == 0:
            st.write(f"1. **{disease_name}** (distance: {dist:.2f}) - This is the predicted disease")
        else:
            st.write(f"{i+1}. {disease_name} (distance: {dist:.2f})")

st.header("Model Information")
st.write(f"""
- **Encoding method**: {encoding_type}
- **Number of neighbors (k)**: {n_neighbors}
- **Distance metric**: {metric}
- **Number of diseases in model**: {len(label_encoder.classes_)}
""")

if st.checkbox("Show sample diseases in model"):
    st.write("Sample of diseases in the model:")
    st.dataframe(disease_features[['Disease', 'Symptoms Text']].head(10))

def extract_category(disease_name):
    categories = {
        'Cardiovascular': ['heart', 'cardiac', 'vascular', 'aortic', 'hypertension', 'stroke'],
        'Respiratory': ['asthma', 'copd', 'pneumonia', 'pulmonary', 'tuberculosis'],
        'Endocrine': ['diabetes', 'thyroid', 'adrenal', 'pituitary'],
        'Neurological': ['alzheimer', 'epilepsy', 'multiple', 'migraine'],
        'Gastrointestinal': ['gastritis', 'ulcer', 'gastrointestinal', 'bleeding']
    }
    
    disease_lower = disease_name.lower()
    for category, keywords in categories.items():
        if any(keyword in disease_lower for keyword in keywords):
            return category
    return 'Other'

disease_features['Category'] = disease_features['Disease'].apply(extract_category)
category_counts = disease_features['Category'].value_counts()

st.header("Disease Categories Distribution")
st.bar_chart(category_counts)

st.markdown("---")
st.markdown("""
**Note**: This is a demonstration application. For actual clinical use, consult with medical professionals.
""")