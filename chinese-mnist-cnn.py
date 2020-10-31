import os, shutil, random
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Avaliable: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.chdir("E:\\chinese-mnist")
base_dir = os.getcwd()
original_picture_dir = "E:\\chinese-mnist\\data"

# 建立訓練、驗證、測試，三個資料夾
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(val_dir):
    os.mkdir(val_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

chinese_mnist_list1 = ["zero", "one", "two", "three", "four",
                      "five", "six", "seven", "eight", "nine",
                      "ten", "hundred", "thousand", "wan", "yi"]
chinese_mnist_list2 = ["零", "一", "二", "三", "四",
                       "五", "六", "七", "八", "九",
                       "十", "百", "千", "萬", "億"]
# 在訓練資料夾中建立個別15個圖片資料夾
# train_name_list = [f"train_{i}_dir" for i in chinese_mnist_list1]
# for k in range(15):
#     globals()[train_name_list[k]]=os.path.join(train_dir, chinese_mnist_list2[k])
#     if not os.path.isdir(train_dir + "\\" + chinese_mnist_list2[k]):
#         os.mkdir(train_dir + "\\" + chinese_mnist_list2[k])

# 在驗證資料夾中建立個別15個圖片資料夾
# valid_name_list = [f"valid_{i}_dir" for i in chinese_mnist_list1]
# for k in range(15):
#     globals()[valid_name_list[k]]=os.path.join(val_dir, chinese_mnist_list2[k])
#     if not os.path.isdir(val_dir + "\\" + chinese_mnist_list2[k]):
#         os.mkdir(val_dir + "\\" + chinese_mnist_list2[k])

# 在測試資料夾中建立個別15個圖片資料夾
# test_name_list = [f"test_{i}_dir" for i in chinese_mnist_list1]
# for k in range(15):
#     globals()[test_name_list[k]]=os.path.join(test_dir, chinese_mnist_list2[k])
#     if not os.path.isdir(test_dir + "\\" + chinese_mnist_list2[k]):
#         os.mkdir(test_dir + "\\" + chinese_mnist_list2[k])

first_nums = [i for i in range(1, 101)]
second_nums = [j for j in range(1, 11)]
third_nums = [k for k in range(1, 16)]


random.shuffle(first_nums)
random.shuffle(second_nums)

# 將原圖片資料中800組圖片複製到訓練資料夾
# for f_num in first_nums:
#     for s_num in second_nums[:8]:
#         for k in third_nums:
#             fname = f"input_{f_num}_{s_num}_{k}.jpg"
#             src = os.path.join(original_picture_dir, fname)
#             dst = os.path.join(train_dir +r"\\"+ chinese_mnist_list2[k-1], fname)
#             shutil.copyfile(src, dst)

# 將原圖片資料中800組圖片複製到驗證資料夾
# for f_num in first_nums:
#     for s_num in second_nums[8:9]:
#         for k in third_nums:
#             fname = f"input_{f_num}_{s_num}_{k}.jpg"
#             src = os.path.join(original_picture_dir, fname)
#             dst = os.path.join(val_dir +r"\\"+ chinese_mnist_list2[k-1], fname)
#             shutil.copyfile(src, dst)

# # 將原圖片資料中800組圖片複製到測試資料夾
# for f_num in first_nums:
#     for s_num in second_nums[9:10]:
#         for k in third_nums:
#             fname = f"input_{f_num}_{s_num}_{k}.jpg"
#             src = os.path.join(original_picture_dir, fname)
#             dst = os.path.join(test_dir +r"\\"+ chinese_mnist_list2[k-1], fname)
#             shutil.copyfile(src, dst)

# 建立卷積神經網路
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(15, activation='softmax')
    ])
model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(64,64),
    batch_size=20, class_mode='sparse',
    color_mode="grayscale"
    )
validation_generator = validation_datagen.flow_from_directory(
    val_dir, target_size=(64,64),
    batch_size=20, class_mode='sparse',
    color_mode="grayscale"
    )
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

history = model.fit_generator(
    train_generator,
    steps_per_epoch=300,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[tensorboard_callback]
    )
model.save('chinese_mnist.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64,64),
    class_mode='sparse',
    color_mode="grayscale"
    )
test_loss, test_acc = model.evaluate(test_generator)
print(test_loss, test_acc)
model.predict(test_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(16,10))
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure(figsize=(16,10))
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
