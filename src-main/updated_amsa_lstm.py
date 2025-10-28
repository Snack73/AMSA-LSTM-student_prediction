# AMSA-LSTM Prediction - Optimized for Small Datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Layer, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("=== AMSA-LSTM: Adaptive Multi-Scale Attention LSTM ===")
print("Optimized for Small Datasets")
print("=" * 60)


# CUSTOM LAYERS (Simplified for small datasets)

class AdaptiveAttentionLayer(Layer):
    """Simplified attention mechanism for small datasets"""
    def __init__(self, **kwargs):
        super(AdaptiveAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Reduced complexity for small datasets
        self.attention_dense = Dense(input_shape[-1] // 2, activation='tanh', 
                                     kernel_regularizer=l2(0.01), name='attention_weights')
        self.attention_softmax = Dense(input_shape[-1], activation='softmax', 
                                       kernel_regularizer=l2(0.01), name='attention_softmax')
        super(AdaptiveAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        attention_weights = self.attention_softmax(attention_scores)
        context_vector = K.sum(inputs * attention_weights, axis=1)
        return [context_vector, attention_weights]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1], input_shape[2])]


# DATA PREPROCESSING (Optimized for small datasets)

def load_and_preprocess_datasets():
    """Enhanced preprocessing with data augmentation for small datasets"""
    print("Loading and preprocessing datasets...")

    try:
        df_mooc = pd.read_csv("/content/data/xAPI-Edu-Data.csv")
        df_comm = pd.read_csv("/content/data/StudentsPerformance.csv")
        df_iit = pd.read_csv("/content/data/academic_performance_dataset_V2.csv")
        print("✓ All datasets loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None

    # FINE-SCALE PROCESSING - Simplified features
    fine_features = ['VisITedResources', 'raisedhands', 'Discussion', 'AnnouncementsView']
    X_fine_raw = df_mooc[fine_features].values
    
    # Add minimal engineered features
    X_fine_enhanced = np.column_stack([
        X_fine_raw,
        np.mean(X_fine_raw, axis=1, keepdims=True),
    ])

    # MEDIUM-SCALE PROCESSING - Simplified
    medium_features = ['math score', 'reading score', 'writing score']
    X_medium_raw = df_comm[medium_features].values
    
    X_medium_enhanced = np.column_stack([
        X_medium_raw,
        np.mean(X_medium_raw, axis=1, keepdims=True),
    ])

    # COARSE-SCALE PROCESSING - Simplified
    semester_cols = ['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400']
    df_iit_clean = df_iit.dropna(subset=semester_cols + ['CGPA'])
    
    X_coarse_raw = df_iit_clean[semester_cols].values
    
    # Add trend feature
    cgpa_trend = np.mean(np.diff(X_coarse_raw, axis=1), axis=1, keepdims=True)
    X_coarse_enhanced = np.column_stack([X_coarse_raw, cgpa_trend])

    y = df_iit_clean['CGPA'].values

    # Align sample sizes
    min_samples = min(len(X_fine_enhanced), len(X_medium_enhanced), len(X_coarse_enhanced), len(y))
    X_fine_enhanced = X_fine_enhanced[:min_samples]
    X_medium_enhanced = X_medium_enhanced[:min_samples]
    X_coarse_enhanced = X_coarse_enhanced[:min_samples]
    y = y[:min_samples]

    # DATA AUGMENTATION for small datasets
    # Add slight noise to create more training examples
    noise_factor = 0.01
    X_fine_aug = X_fine_enhanced + np.random.normal(0, noise_factor, X_fine_enhanced.shape)
    X_medium_aug = X_medium_enhanced + np.random.normal(0, noise_factor, X_medium_enhanced.shape)
    X_coarse_aug = X_coarse_enhanced + np.random.normal(0, noise_factor, X_coarse_enhanced.shape)
    
    # Combine original and augmented data
    X_fine_enhanced = np.vstack([X_fine_enhanced, X_fine_aug])
    X_medium_enhanced = np.vstack([X_medium_enhanced, X_medium_aug])
    X_coarse_enhanced = np.vstack([X_coarse_enhanced, X_coarse_aug])
    y = np.hstack([y, y])

    # Scale features
    scaler_fine = MinMaxScaler()
    scaler_medium = MinMaxScaler()
    scaler_coarse = MinMaxScaler()

    X_fine_scaled = scaler_fine.fit_transform(X_fine_enhanced)
    X_medium_scaled = scaler_medium.fit_transform(X_medium_enhanced)
    X_coarse_scaled = scaler_coarse.fit_transform(X_coarse_enhanced)

    # Reshape for LSTM
    X_fine = X_fine_scaled.reshape(-1, X_fine_scaled.shape[1], 1)
    X_medium = X_medium_scaled.reshape(-1, X_medium_scaled.shape[1], 1)
    X_coarse = X_coarse_scaled.reshape(-1, X_coarse_scaled.shape[1], 1)

    print(f"✓ Processed {len(y)} samples (with augmentation)")
    print(f"  Fine-scale shape: {X_fine.shape}")
    print(f"  Medium-scale shape: {X_medium.shape}")
    print(f"  Coarse-scale shape: {X_coarse.shape}")

    return X_fine, X_medium, X_coarse, y


# MODEL ARCHITECTURE (Reduced complexity for small datasets)

def build_fine_scale_branch(input_shape):
    """Reduced capacity fine-scale branch"""
    input_layer = Input(shape=input_shape, name='fine_input')
    
    # Smaller LSTM with regularization
    lstm_out = LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01), 
                    recurrent_regularizer=l2(0.01), name='fine_lstm')(input_layer)
    lstm_out = BatchNormalization(name='fine_bn1')(lstm_out)
    
    context_vector, attention_weights = AdaptiveAttentionLayer(name='fine_attention')(lstm_out)
    
    # Simplified dense layers
    dense_out = Dense(16, activation='relu', kernel_regularizer=l2(0.01), name='fine_dense1')(context_vector)
    dense_out = Dropout(0.3, name='fine_dropout1')(dense_out)
    
    return input_layer, dense_out, attention_weights

def build_medium_scale_branch(input_shape):
    """Reduced capacity medium-scale branch"""
    input_layer = Input(shape=input_shape, name='medium_input')
    
    lstm_out = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01),
                    recurrent_regularizer=l2(0.01), name='medium_lstm')(input_layer)
    lstm_out = BatchNormalization(name='medium_bn1')(lstm_out)
    
    context_vector, attention_weights = AdaptiveAttentionLayer(name='medium_attention')(lstm_out)
    
    dense_out = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='medium_dense1')(context_vector)
    dense_out = Dropout(0.3, name='medium_dropout1')(dense_out)
    
    return input_layer, dense_out, attention_weights

def build_coarse_scale_branch(input_shape):
    """Reduced capacity coarse-scale branch"""
    input_layer = Input(shape=input_shape, name='coarse_input')
    
    lstm_out = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01),
                    recurrent_regularizer=l2(0.01), name='coarse_lstm')(input_layer)
    lstm_out = BatchNormalization(name='coarse_bn1')(lstm_out)
    
    context_vector, attention_weights = AdaptiveAttentionLayer(name='coarse_attention')(lstm_out)
    
    dense_out = Dense(64, activation='relu', kernel_regularizer=l2(0.01), name='coarse_dense1')(context_vector)
    dense_out = Dropout(0.2, name='coarse_dropout1')(dense_out)
    
    return input_layer, dense_out, attention_weights

def build_amsa_lstm_model(fine_shape, medium_shape, coarse_shape):
    """Build streamlined AMSA-LSTM architecture"""
    print("Building AMSA-LSTM architecture...")

    fine_input, fine_output, fine_attention = build_fine_scale_branch(fine_shape)
    medium_input, medium_output, medium_attention = build_medium_scale_branch(medium_shape)
    coarse_input, coarse_output, coarse_attention = build_coarse_scale_branch(coarse_shape)

    # Multi-scale fusion
    concatenated = Concatenate(name='multi_scale_fusion')([fine_output, medium_output, coarse_output])

    # Simplified fusion layers
    fusion_dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01), name='fusion_dense1')(concatenated)
    fusion_dense = BatchNormalization(name='fusion_bn')(fusion_dense)
    fusion_dense = Dropout(0.3, name='fusion_dropout1')(fusion_dense)

    fusion_dense = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='fusion_dense2')(fusion_dense)
    fusion_dense = Dropout(0.2, name='fusion_dropout2')(fusion_dense)

    # Output layer
    prediction = Dense(1, activation='linear', name='output')(fusion_dense)

    # Create models
    training_model = Model(
        inputs=[fine_input, medium_input, coarse_input],
        outputs=prediction,
        name='AMSA_LSTM_Training'
    )

    full_model = Model(
        inputs=[fine_input, medium_input, coarse_input],
        outputs=[prediction, fine_attention, medium_attention, coarse_attention],
        name='AMSA_LSTM_Full'
    )

    print("✓ AMSA-LSTM architecture built successfully")
    return training_model, full_model


# TRAINING UTILITIES

def temporal_train_test_split(X_fine, X_medium, X_coarse, y, test_size=0.2, val_size=0.1):
    """Temporal split optimized for small datasets"""
    n_samples = len(y)
    
    train_end = int(n_samples * (1 - test_size - val_size))
    val_end = int(n_samples * (1 - test_size))

    X_fine_train, X_fine_val, X_fine_test = X_fine[:train_end], X_fine[train_end:val_end], X_fine[val_end:]
    X_medium_train, X_medium_val, X_medium_test = X_medium[:train_end], X_medium[train_end:val_end], X_medium[val_end:]
    X_coarse_train, X_coarse_val, X_coarse_test = X_coarse[:train_end], X_coarse[train_end:val_end], X_coarse[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    return (X_fine_train, X_medium_train, X_coarse_train, y_train,
            X_fine_val, X_medium_val, X_coarse_val, y_val,
            X_fine_test, X_medium_test, X_coarse_test, y_test)


# BASELINE MODELS

def build_baseline_models(X_fine_train, X_medium_train, X_coarse_train, y_train):
    """Build baseline models"""
    print("Training baseline models...")

    X_train_flat = np.concatenate([
        X_fine_train.reshape(X_fine_train.shape[0], -1),
        X_medium_train.reshape(X_medium_train.shape[0], -1),
        X_coarse_train.reshape(X_coarse_train.shape[0], -1)
    ], axis=1)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_flat, y_train)

    # Random Forest with reduced complexity
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train_flat, y_train)

    # Standard LSTM (reduced capacity)
    X_combined = np.concatenate([X_fine_train, X_medium_train, X_coarse_train], axis=1)

    lstm_input = Input(shape=(X_combined.shape[1], X_combined.shape[2]))
    lstm_out = LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))(lstm_input)
    lstm_out = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_output = Dense(1, activation='linear')(lstm_out)

    standard_lstm = Model(inputs=lstm_input, outputs=lstm_output)
    standard_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    print("✓ Baseline models created")
    return lr_model, rf_model, standard_lstm, X_train_flat, X_combined

# EVALUATION METRICS

def calculate_comprehensive_metrics(y_true, y_pred, model_name="Model"):
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    # Statistical significance test
    t_stat, p_value = stats.ttest_rel(y_true.flatten(), y_pred.flatten())

    print(f"\n{model_name} Performance Metrics:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  P-value: {p_value:.6f}")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Correlation': correlation, 'P-value': p_value}


# MAIN EXECUTION

def main():
    """Main execution function"""
    print("Starting AMSA-LSTM experiment...")

    # Load and preprocess data
    X_fine, X_medium, X_coarse, y = load_and_preprocess_datasets()
    if X_fine is None:
        return

    # Split data
    print("\nPerforming temporal train-test-validation split...")
    (X_fine_train, X_medium_train, X_coarse_train, y_train,
     X_fine_val, X_medium_val, X_coarse_val, y_val,
     X_fine_test, X_medium_test, X_coarse_test, y_test) = temporal_train_test_split(
        X_fine, X_medium, X_coarse, y, test_size=0.2, val_size=0.1)

    print(f"✓ Data split completed:")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")

    # Build AMSA-LSTM model
    training_model, full_model = build_amsa_lstm_model(
        X_fine.shape[1:], X_medium.shape[1:], X_coarse.shape[1:])

    # Compile with higher learning rate for small datasets
    training_model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='mse',
        metrics=['mae']
    )

    print(f"\nAMSA-LSTM Architecture Summary:")
    training_model.summary()

    # Callbacks with adjusted patience
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    ]

    # Train AMSA-LSTM with more epochs
    print("\nTraining AMSA-LSTM model...")
    history = training_model.fit(
        [X_fine_train, X_medium_train, X_coarse_train], y_train,
        validation_data=([X_fine_val, X_medium_val, X_coarse_val], y_val),
        epochs=100,
        batch_size=16,  # Smaller batch size for small datasets
        callbacks=callbacks,
        verbose=1
    )

    # Build and train baseline models
    lr_model, rf_model, standard_lstm, X_train_flat, X_combined_train = build_baseline_models(
        X_fine_train, X_medium_train, X_coarse_train, y_train)

    # Train standard LSTM
    X_combined_val = np.concatenate([X_fine_val, X_medium_val, X_coarse_val], axis=1)
    standard_lstm.fit(X_combined_train, y_train,
                     validation_data=(X_combined_val, y_val),
                     epochs=50, batch_size=16, verbose=0)

    # Make predictions
    print("\nMaking predictions...")

    # AMSA-LSTM predictions
    amsa_pred = training_model.predict([X_fine_test, X_medium_test, X_coarse_test])

    # Get attention weights from full model
    full_predictions = full_model.predict([X_fine_test, X_medium_test, X_coarse_test])
    amsa_pred_full, fine_attn, medium_attn, coarse_attn = full_predictions

    # Baseline predictions
    X_test_flat = np.concatenate([
        X_fine_test.reshape(X_fine_test.shape[0], -1),
        X_medium_test.reshape(X_medium_test.shape[0], -1),
        X_coarse_test.reshape(X_coarse_test.shape[0], -1)
    ], axis=1)

    lr_pred = lr_model.predict(X_test_flat)
    rf_pred = rf_model.predict(X_test_flat)

    X_combined_test = np.concatenate([X_fine_test, X_medium_test, X_coarse_test], axis=1)
    lstm_pred = standard_lstm.predict(X_combined_test)

    # Calculate comprehensive metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE EVALUATION")
    print("="*60)

    results = {}
    results['AMSA-LSTM'] = calculate_comprehensive_metrics(y_test, amsa_pred, "AMSA-LSTM")
    results['Standard LSTM'] = calculate_comprehensive_metrics(y_test, lstm_pred, "Standard LSTM")
    results['Random Forest'] = calculate_comprehensive_metrics(y_test, rf_pred, "Random Forest")
    results['Linear Regression'] = calculate_comprehensive_metrics(y_test, lr_pred, "Linear Regression")

    # Calculate improvement
    amsa_mae = results['AMSA-LSTM']['MAE']
    lstm_mae = results['Standard LSTM']['MAE']
    improvement = ((lstm_mae - amsa_mae) / lstm_mae) * 100

    print(f"\n🎯 AMSA-LSTM achieves {improvement:.1f}% improvement over Standard LSTM")

    return history, results, y_test, amsa_pred, lstm_pred, rf_pred, lr_pred

# Run the experiment
history, results, y_test, amsa_pred, lstm_pred, rf_pred, lr_pred = main()

# Print final results in JSON format
print("\n" + "="*60)
print("FINAL RESULTS (JSON FORMAT)")
print("="*60)
import json
print(json.dumps(results, indent=2))

# Vizualization 
import matplotlib.pyplot as plt
import pandas as pd

# Ensure the results variable is available, assuming it was generated by running the previous cells
if 'results' in locals():
    # Convert results to a pandas DataFrame for easier plotting
    results_df = pd.DataFrame(results).T

    # Define metrics to plot
    metrics_to_plot = ['MAE', 'RMSE', 'R2', 'Correlation']

    # Create bar charts for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        results_df[metric].plot(kind='bar', ax=axes[i], color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[i].set_title(f'{metric} Comparison', fontsize=14)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', linestyle='--')

    plt.tight_layout()
    plt.show()

    # You can also display the results table
    print("\nPerformance Metrics Table:")
    display(results_df)

else:
    print("Please run the cell that calls the main() function to generate the 'results' variable before plotting.")

# Plot training history if available (assuming 'history' variable exists)
if 'history' in locals() and history is not None:
    print("\nAMSA-LSTM Training History:")
    history_df = pd.DataFrame(history.history)

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('AMSA-LSTM Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot MAE
    if 'mae' in history_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['mae'], label='Training MAE')
        plt.plot(history_df['val_mae'], label='Validation MAE')
        plt.title('AMSA-LSTM Training and Validation MAE Over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)object
        plt.ylabel('MAE', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("MAE history not available in the training logs.")
else:
    print("\nAMSA-LSTM training history not available. Please ensure the main() function was run and returned the history .")
