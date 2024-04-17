from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model("D:/ML/rockpaperscissors_model.h5")

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Contoh: Load data tambahan menggunakan ImageDataGenerator
additional_data_generator = ImageDataGenerator(rescale=1./255,
                                               validation_split=0.2)

additional_generator = additional_data_generator.flow_from_directory(
    "path_to_additional_data_directory",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

model.fit(
    additional_generator,
    steps_per_epoch=additional_generator.samples // additional_generator.batch_size,
    epochs=5  # Ganti dengan jumlah epoch yang diinginkan
)
