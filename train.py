import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from PokeBNN import PokeBNN
from ReActNet import ReActNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = PokeBNN(num_classes=10).build(input_shape=(32, 32, 3))
# stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2
# model = ReActNet(stage_out_channel, num_classes=10).build(input_shape=(32, 32, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, batch_size=128, validation_data=(test_images, test_labels))
print(history.history.keys())
model.save('./train_result/Poke_no_shrink_remove.h5')

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig("./train_result/Poke_no_shrink_remove_acc.png")
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig("./train_result/Poke_no_shrink_remove_loss.png")
plt.close()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

