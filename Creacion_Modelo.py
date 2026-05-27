# =========================================================
# Ajuste y evaluación final del modelo de clasificación de gatos, perros y otros
# =========================================================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================================================
# RUTAS
# =========================================================
train_dir = r"dataset_dividido\train"
val_dir   = r"dataset_dividido\valid"
test_dir  = r"dataset_dividido\test"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15

# =========================================================
# MODELO (MobileNetV2 + ajuste)
# =========================================================
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128)(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(3, activation='softmax')(x)   # 3 clases

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =========================================================
# DATA GENERATORS
# =========================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================================================
# ENTRENAMIENTO
# =========================================================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================================================
# EVALUACIÓN COMPLETA
# =========================================================
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

f1 = f1_score(y_true, y_pred, average='weighted')
acc = accuracy_score(y_true, y_pred)
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")

# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_gen.class_indices.keys(),
            yticklabels=test_gen.class_indices.keys())
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.show()

# =========================================================
# CURVAS DE APRENDIZAJE
# =========================================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión (ajuste)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (ajuste)')
plt.legend()
plt.show()

# =========================================================
# GUARDAR MODELO
# =========================================================
model.save("modelo_gatos_perros_otros_final.h5")
print("Modelo ajustado guardado como modelo_gatos_perros_otros_final.h5")
