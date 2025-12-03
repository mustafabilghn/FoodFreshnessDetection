from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    def __init__(self, data_dir="data", img_size=(224, 224), batch_size=16, seed=42):  # batch size 32->16
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed

    def load_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )

        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2
        )

        train_data = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=self.seed,
        )

        val_data = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=self.seed,
        )

        return train_data, val_data


if __name__ == "__main__":
    loader = DataLoader(data_dir="../data")
    train_data, val_data = loader.load_data()

    print("Train:", train_data.class_indices)
    print("Test:", val_data.class_indices)
