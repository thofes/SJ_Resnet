import tensorflow as tf
print("dataloader module imported successfully")
def load_data(train_dir, val_dir, image_size=(270,480), batch_size=50):
    # Load the data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        subset="training",
        validation_split = 0.2,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        subset="validation",
        validation_split = 0.2,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names