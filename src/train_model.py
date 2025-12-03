from preprocess import DataLoader
from model_cnn import create_model
import matplotlib.pyplot as plt

loader = DataLoader(data_dir="../data")
train_data, val_data = loader.load_data()

model = create_model()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=6,
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

model.save("model/food_freshness_model.keras")
print("Model Trained and Saved Successfully")
