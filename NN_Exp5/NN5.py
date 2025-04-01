#93.67
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt

# Load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Expand dimensions for CNN input
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Function to create a VGG-like CNN model
def model_arc(filter_size=3, optimizer='adam', reg=None):
    model = keras.Sequential([
        layers.Conv2D(64, (filter_size, filter_size), activation='relu', padding='same', input_shape=(28,28,1), kernel_regularizer=reg),
        layers.Conv2D(64, (filter_size, filter_size), activation='relu', padding='same', kernel_regularizer=reg),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),

        layers.Conv2D(128, (filter_size, filter_size), activation='relu', padding='same', kernel_regularizer=reg),
        layers.Conv2D(128, (filter_size, filter_size), activation='relu', padding='same', kernel_regularizer=reg),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),

        layers.Conv2D(256, (filter_size, filter_size), activation='relu', padding='same', kernel_regularizer=reg),
        layers.Conv2D(256, (filter_size, filter_size), activation='relu', padding='same', kernel_regularizer=reg),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),

        layers.Conv2D(512, (filter_size, filter_size), activation='relu', padding='same', kernel_regularizer=reg),
        layers.Conv2D(512, (filter_size, filter_size), activation='relu', padding='same', kernel_regularizer=reg),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Hyperparameter options
batch_sizes = [128]
filter_sizes = [3]
optimizers = ['adam']
regularizations = [None]

# Loop through hyperparameter combinations
for batch_size in batch_sizes:
    for filter_size in filter_sizes:
        for optimizer in optimizers:
            for reg in regularizations:
                reg_name = 'L2(0.001)' if reg else 'None'
                opt_name = 'SGD(momentum=0.9)' if isinstance(optimizer, keras.optimizers.SGD) else optimizer.upper()
                print(f"Training with Batch Size: {batch_size}, Filter Size: {filter_size}, Optimizer: {opt_name}, Regularization: {reg_name}\n")
                
                model = model_arc(filter_size=filter_size, optimizer=optimizer, reg=reg)
                history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
                
                # Evaluate the model
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
                
                # Plot accuracy and loss curves
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Train Accuracy')
                plt.plot(history.history['val_accuracy'], label='Val Accuracy')
                plt.legend()
                plt.title(f'Accuracy (Batch={batch_size}, Filter={filter_size}, Opt={opt_name}, Reg={reg_name})')
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.legend()
                plt.title(f'Loss (Batch={batch_size}, Filter={filter_size}, Opt={opt_name}, Reg={reg_name})')
                
                plt.show()
