import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

##import zipfile
##
##local_zip = '/rockpaperscissors.zip'
##zip_ref = zipfile.ZipFile(local_zip, 'r')
##zip_ref.extractall('/tmp')
##zip_ref.close()

# Menentukan path dataset
base_dir = 'rockpaperscissors'  # Sesuaikan dengan struktur folder Anda

# Membuat direktori untuk data training dan data validasi
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# Memecah data menjadi train set dan validation set
image_files = os.listdir(base_dir)
train_files, val_files = train_test_split(image_files, test_size=0.4, random_state=42)

# Membuat direktori baru untuk data training dan data validasi
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Menyalin data training ke direktori baru
for file in train_files:
    src_path = os.path.join(base_dir, file)
    dest_path = os.path.join(train_dir, file)
    shutil.copy(src_path, dest_path)

# Menyalin data validasi ke direktori baru
for file in val_files:
    src_path = os.path.join(base_dir, file)
    dest_path = os.path.join(validation_dir, file)
    shutil.copy(src_path, dest_path)

# Inisialisasi ImageDataGenerator untuk augmentasi gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Inisialisasi ImageDataGenerator untuk data validasi (tanpa augmentasi)
val_datagen = ImageDataGenerator(rescale=1./255)

# Menggunakan flow_from_directory untuk memuat data training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Menggunakan flow_from_directory untuk memuat data validasi
val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Membangun model sequential
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# Mengkompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.n//val_generator.batch_size
)

# Menyimpan model
model.save('rock_paper_scissors_model.h5')


### Menentukan path dataset
##base_dir = '/rockpaperscissors' 
##
### Membuat direktori untuk data training dan data validasi
##train_dir = os.path.join(base_dir, 'train')
##validation_dir = os.path.join(base_dir, 'val')
##
### Memecah data menjadi train set dan validation set
##image_files = os.listdir(os.path.join(base_dir, 'rps-cv-images'))
##train_files, val_files = train_test_split(image_files, test_size=0.4, random_state=42)
##
### Membuat direktori baru untuk data training dan data validasi
##os.makedirs(train_dir, exist_ok=True)
##os.makedirs(validation_dir, exist_ok=True)
##
### Menyalin data training ke direktori baru
##for file in train_files:
##    src_path = os.path.join(base_dir, 'rps-cv-images', file)
##    dest_path = os.path.join(train_dir, file)
##    shutil.copy(src_path, dest_path)
##
### Menyalin data validasi ke direktori baru
##for file in val_files:
##    src_path = os.path.join(base_dir, 'rps-cv-images', file)
##    dest_path = os.path.join(validation_dir, file)
##    shutil.copy(src_path, dest_path)
##
### Inisialisasi ImageDataGenerator untuk augmentasi gambar
##train_datagen = ImageDataGenerator(
##    rescale=1./255,
##    rotation_range=20,
##    shear_range=0.2,
##    zoom_range=0.2,
##    horizontal_flip=True,
##    fill_mode='nearest'
##)
##
### Inisialisasi ImageDataGenerator untuk data validasi (tanpa augmentasi)
##val_datagen = ImageDataGenerator(rescale=1./255)
##
### Menggunakan flow_from_directory untuk memuat data training
##train_generator = train_datagen.flow_from_directory(
##    train_dir,
##    target_size=(150, 150),
##    batch_size=32,
##    class_mode='categorical'
##)
##
### Menggunakan flow_from_directory untuk memuat data validasi
##val_generator = val_datagen.flow_from_directory(
##    validation_dir,
##    target_size=(150, 150),
##    batch_size=32,
##    class_mode='categorical'
##)
##
### Membangun model sequential
##model = Sequential([
##    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
##    MaxPooling2D(2, 2),
##    Conv2D(64, (3,3), activation='relu'),
##    MaxPooling2D(2,2),
##    Conv2D(128, (3,3), activation='relu'),
##    MaxPooling2D(2,2),
##    Flatten(),
##    Dense(512, activation='relu'),
##    Dense(3, activation='softmax')
##])
##
### Mengkompilasi model
##model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
##
### Melatih model
##history = model.fit(
##    train_generator,
##    steps_per_epoch=train_generator.n//train_generator.batch_size,
##    epochs=10,
##    validation_data=val_generator,
##    validation_steps=val_generator.n//val_generator.batch_size
##)
##
### Menyimpan model
##model.save('rock_paper_scissors_model.h5')
