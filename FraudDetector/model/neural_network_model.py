import sys
import os

# Add the parent directory (C:\AI\FraudDetector) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from data_preparation import load_and_preprocess_data
from sklearn.utils.class_weight import compute_class_weight

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def neural_network_model():
    # Load data
    X, y = load_and_preprocess_data()

    # Split the data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y_train)) - 1))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Standardize the features
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
    class_weight_dict = dict(enumerate(class_weights))

    # Build the neural network model
    model_nn = Sequential([
        Dense(128, input_shape=(X_train_resampled.shape[1],)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(32),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(3, activation='softmax')
    ])

    # Compile the model with a lower learning rate
    model_nn.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

    # Train the model with class weights
    print("\n--- Neural Network Model with Keras ---")
    model_nn.fit(X_train_resampled, y_train_resampled, epochs=100, batch_size=16,
                 validation_split=0.2, callbacks=[early_stopping, reduce_lr],
                 class_weight=class_weight_dict)

    # Evaluate the model on the test set
    y_pred_nn = np.argmax(model_nn.predict(X_test), axis=1)
    report_nn = classification_report(y_test, y_pred_nn, target_names=['Honest', 'Suspicious', 'Fraud'], labels=[0, 1, 2], zero_division=0)
    print("\nFinal Model Evaluation (Neural Network):")
    print(report_nn)

    # Get the absolute path to the 'models' directory
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

    # Save the trained neural network model
    model_nn.save(os.path.join(models_dir, 'neural_network_model.keras'))
    print(f"Neural network model saved as '{os.path.join(models_dir, 'neural_network_model.keras')}'.")

if __name__ == '__main__':
    neural_network_model()
