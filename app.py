from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load all models at startup
try:
    # tfidf = joblib.load('tfidf')
    # scaler = joblib.load('standard_scaler')

    tfidf = joblib.load('tfidf_svm')
    model = joblib.load('svm')

    # pca = joblib.load('pca_model_rf')
    # model = joblib.load('rf_pca')

    # pca = joblib.load('pca_model')
    # model = joblib.load('stacking_model')
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request")  # Debug print
        
        # Get input text
        data = request.get_json(force=True)  # Add force=True to handle malformed requests
        print(f"Received data: {data}")  # Debug print
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided', 'status': 'error'}), 400
        
        text = data['text']
        print(f"Processing text: {text}")  # Debug print

        # 1. TF-IDF Transformation
        tfidf_features = tfidf.transform([text]).toarray()
        print(f"TF-IDF features shape: {tfidf_features.shape}")  # Debug print

        print("TF-IDF DONE")

        # # 2. Standard Scaling
        # scaled_features = scaler.transform(dense_features)
        # print(f"Scaled features shape: {scaled_features.shape}")  # Debug print

        print("SCALING DONE")

        # # 3. Apply PCA
        # pca_features = pca.transform(dense_features)
        # print(f"PCA features shape: {pca_features.shape}")  # Debug print

        # print("PCA DONE")

        # 4. Predict
        prediction = model.predict(tfidf_features)
        print(f"Raw prediction: {prediction}")  # Debug print

        # 5. Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(tfidf_features)[0].tolist()
            print(f"Probabilities: {probabilities}")  # Debug print

        return jsonify({
            "prediction": int(prediction[0]),
            "probabilities": probabilities,
            "status": "success"
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}", flush=True)  # Ensure error prints immediately
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)