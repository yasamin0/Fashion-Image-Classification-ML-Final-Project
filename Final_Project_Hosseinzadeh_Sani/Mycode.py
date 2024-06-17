import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the data
train_data = np.load('clothes/train.npz')
Xtrain, Ytrain = train_data['images'], train_data['labels']

test_data = np.load('clothes/test.npz')
Xtest, Ytest = test_data['images'], test_data['labels']

# Print basic statistics
print(f'Training data shape: {Xtrain.shape}')
print(f'Training labels shape: {Ytrain.shape}')
print(f'Test data shape: {Xtest.shape}')
print(f'Test labels shape: {Ytest.shape}')

# Number of samples per class
classes, counts = np.unique(Ytrain, return_counts=True)
for cls, count in zip(classes, counts):
    print(f'Class {cls}: {count} samples')

# Visualize some samples
def plot_samples(X, y, classes, samples_per_class=10):
    num_classes = len(classes)
    for cls in range(num_classes):
        idxs = np.flatnonzero(y == cls)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + cls + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[idx], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(classes[cls])
    plt.show()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plot_samples(Xtrain, Ytrain, class_names)

# Preprocess the data
Xtrain = Xtrain.reshape(-1, 28, 28, 1) / 255.0
Xtest = Xtest.reshape(-1, 28, 28, 1) / 255.0
Ytrain = to_categorical(Ytrain, num_classes=10)
Ytest = to_categorical(Ytest, num_classes=10)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10, batch_size=128)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(Xtest, Ytest)
print(f'Test accuracy: {test_accuracy:.4f}')

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_training_history(history)

# Confusion matrix
Ypred = model.predict(Xtest).argmax(axis=1)
Ytrue = Ytest.argmax(axis=1)
cm = confusion_matrix(Ytrue, Ypred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
