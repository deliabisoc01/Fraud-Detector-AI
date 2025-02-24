from flask import Flask, request, render_template, flash, redirect
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report
from data_preparation import load_and_preprocess_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Load the trained models
def load_models():
    """Load the machine learning models."""
    try:
        logistic_model = joblib.load('models/logistic_regression_model.pkl')
        neural_network_model = tf.keras.models.load_model('models/neural_network_model.keras')
        return logistic_model, neural_network_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

logistic_model, neural_network_model = load_models()

@app.route('/', methods=['GET', 'POST'])
def index():
    evaluation_report = None
    result_html = None

    if request.method == 'POST':
        file = request.files.get('file')
        model_choice = request.form.get('model_choice')

        # Check if file and model choice are provided
        if not file or not model_choice:
            flash('Please upload a file and select a model.', 'danger')
            return redirect(request.url)

        # Process the uploaded file
        transactions_df = process_uploaded_file(file)
        if transactions_df is None:
            flash('Invalid file format. Please upload a valid Excel file.', 'danger')
            return redirect(request.url)

        # Preprocess the data and make predictions
        try:
            X = load_and_preprocess_data(transactions_df, is_file=False)
            predictions, evaluation_report = make_predictions(X, model_choice)
            
            # Add predictions to the DataFrame
            transactions_df['Prediction'] = map_predictions_to_labels(predictions)
            
            # Convert the DataFrame to HTML
            result_html = transactions_df.to_html(classes='table table-striped', index=False)
        except Exception as e:
            flash(f'Error processing the data: {e}', 'danger')
            return redirect(request.url)

    # Render the template with the table and evaluation report
    return render_template('index.html', table=result_html, evaluation_report=evaluation_report)

@app.route('/presentation')
def presentation():
    """Route to render the presentation page."""
    return render_template('presentation.html')

def process_uploaded_file(file):
    """Read the uploaded file into a DataFrame."""
    try:
        transactions_df = pd.read_excel(file, sheet_name='Transactions')
        return transactions_df
    except Exception as e:
        print(f"Error reading the uploaded file: {e}")
        return None

def make_predictions(X, model_choice):
    """Make predictions using the selected model."""
    if model_choice == 'Logistic Regression' and logistic_model:
        predictions = logistic_model.predict(X)
        y_true = [0] * len(predictions)  # Dummy y_true for classification report
        evaluation_report = classification_report(
            y_true, predictions, target_names=['Honest', 'Suspicious', 'Fraud'], zero_division=0
        )
    elif model_choice == 'Neural Network' and neural_network_model:
        predictions = neural_network_model.predict(X).argmax(axis=1)
        y_true = [0] * len(predictions)  # Dummy y_true for classification report
        evaluation_report = classification_report(
            y_true, predictions, target_names=['Honest', 'Suspicious', 'Fraud'], zero_division=0
        )
    else:
        predictions = []
        evaluation_report = None

    return predictions, evaluation_report

def map_predictions_to_labels(predictions):
    """Map numeric predictions to their corresponding labels."""
    label_mapping = {0: 'Honest', 1: 'Suspicious', 2: 'Fraud'}
    return [label_mapping.get(pred, 'Unknown') for pred in predictions]

if __name__ == '__main__':
    app.run(debug=True, port=8080)
