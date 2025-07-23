# Model: MobileNetv2

# Learning Rate: Adaptive

# Batch Size: 8

# Max EPOCH = 100

# Patience = 15

# Time Taken = 

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# --- ADDED for classification report and confusion matrix ---
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- 1. Configuration and Setup ---
PROCESSED_DIR = os.path.join('processed_data', 'BrinjalFruitX')
RESULTS_DIR = 'results'
MODEL_NAME = 'mobilenetv2_bs8_adaptive_lr'

model_results_dir = os.path.join(RESULTS_DIR, MODEL_NAME)
os.makedirs(model_results_dir, exist_ok=True)

print("Loading preprocessed data...")
# --- Data loading ---
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
with open(os.path.join(PROCESSED_DIR, 'class_names.json'), 'r') as f:
    class_names = json.load(f)
print("Data loaded successfully.")

# --- 2. Define and Train the Model (MobileNetV2) ---
EPOCHS = 100
BATCH_SIZE = 8

# --- Learning Rate Configuration & Callbacks ---
INITIAL_LR = 0.0001
DECAY_RATE = 0.96
DECAY_STEPS = 1000

PATIENCE = 15

def lr_scheduler(epoch, lr):
    return INITIAL_LR * DECAY_RATE**(epoch * (X_train.shape[0] // BATCH_SIZE) / DECAY_STEPS)

early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, restore_best_weights=True)
checkpoint_path = os.path.join(model_results_dir, 'best_model.keras')
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)
callbacks_list = [early_stopping, model_checkpoint, lr_callback]

# --- Model Definition ---
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
base_model.trainable = True
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"\nStarting {MODEL_NAME} model training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks_list
)
print("Model training complete.")


# --- 3. Visualize Performance and Save Figure ---
# --- MODIFIED: Merged into a single plot to match the paper's style ---
print("\nGenerating and saving single performance plot...")
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'{MODEL_NAME} Performance')
plt.ylabel('Accuracy & Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig(os.path.join(model_results_dir, 'performance_plot.png'))
plt.show()


# --- 4. In-Depth Evaluation and Save Results ---
# --- ADDED: Detailed report for the VALIDATION set ---
print(f"\n--- {MODEL_NAME} Validation Set Evaluation ---")
y_pred_val_probs = model.predict(X_val)
y_pred_val_classes = np.argmax(y_pred_val_probs, axis=1)

print("\nValidation Set Classification Report:\n")
print(classification_report(y_val, y_pred_val_classes, target_names=class_names))

# --- Evaluation on the TEST set remains the same ---
print(f"\n--- {MODEL_NAME} Final Test Set Evaluation ---")
y_pred_test_probs = model.predict(X_test)
y_pred_test_classes = np.argmax(y_pred_test_probs, axis=1)

report_dict = classification_report(y_test, y_pred_test_classes, target_names=class_names, output_dict=True)
print("\nTest Set Classification Report:\n")
print(classification_report(y_test, y_pred_test_classes, target_names=class_names))

# --- Saving the Test Set report and Confusion Matrix ---
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(model_results_dir, 'classification_report.csv'))
print(f"Test classification report saved to {os.path.join(model_results_dir, 'classification_report.csv')}")

cm = confusion_matrix(y_test, y_pred_test_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_NAME} Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(model_results_dir, 'confusion_matrix.png'))
plt.show()

# --- 5. Update Summary Results File ---
# This section remains the same
print("\nUpdating summary results file...")
summary_file = os.path.join(RESULTS_DIR, 'summary_results.csv')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

summary_data = {
    'model_name': MODEL_NAME,
    'test_accuracy': f"{test_accuracy:.4f}",
    'test_loss': f"{test_loss:.4f}",
    'macro_avg_f1-score': f"{report_dict['macro avg']['f1-score']:.4f}",
    'weighted_avg_f1-score': f"{report_dict['weighted avg']['f1-score']:.4f}"
}
new_results_df = pd.DataFrame([summary_data])

if os.path.exists(summary_file):
    summary_df = pd.read_csv(summary_file)
    if MODEL_NAME in summary_df['model_name'].values:
        summary_df.loc[summary_df['model_name'] == MODEL_NAME] = new_results_df.iloc[0].values
    else:
        summary_df = pd.concat([summary_df, new_results_df], ignore_index=True)
    summary_df.to_csv(summary_file, index=False)
    print(f"Updated results for {MODEL_NAME} in {summary_file}")
else:
    new_results_df.to_csv(summary_file, index=False)
    print(f"Created new summary results file at {summary_file}")


# --- 6. Save the Trained Model ---
print(f"\nThe best version of {MODEL_NAME} was saved to '{checkpoint_path}'")