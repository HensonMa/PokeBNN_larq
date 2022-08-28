from model import PokeBNN, ResNet50
import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import larq as lq
from tqdm import tqdm
import os

TOTAL_EPOCH = 200
LOSS_OBJECT = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
BATCH_SIZE = 16
LR = 0.01
DECAY_STEP = 100

learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=LR, decay_steps=DECAY_STEP, end_learning_rate=LR/10)
OPTIMIZER = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.99, learning_rate=learning_rate_fn)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def dataset_fn_train(ds):
    seed = int(tf.random.uniform(shape=(1,), minval=0, maxval=100)[0])
    ds = ds.map(lambda a, b: tuple([tf.image.resize(a, size=[34, 34]), b]))
    ds = ds.map(lambda a, b: tuple([tf.image.random_crop(a, size=(32, 32, 3), seed=seed), b]))
    ds = ds.map(lambda a, b: tuple([tf.image.random_flip_left_right(a, seed=seed), b]))
    ds = ds.map(lambda a, b: tuple([tf.image.resize(a, size=[224, 224]), b]))
    ds = ds.shuffle(50000)
    ds = ds.batch(BATCH_SIZE)
    return ds

def dataset_fn_test(ds):
    ds = ds.map(lambda a, b: tuple([tf.image.resize(a, size=[224, 224]), b]))
    ds = ds.shuffle(10000)
    ds = ds.batch(BATCH_SIZE)
    return ds


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    train_images = tf.image.per_image_standardization(train_images)
    test_images = tf.image.per_image_standardization(test_images)

    def gen_train():
        for (i, j) in list(zip(train_images, train_labels)):
            yield i, j

    def gen_test():
        for (i, j) in list(zip(test_images, test_labels)):
            yield i, j

    train_set = tf.data.Dataset.from_generator(gen_train, output_signature=(tf.TensorSpec(shape=(32, 32, 3)), tf.TensorSpec(shape=(1,))))
    test_set = tf.data.Dataset.from_generator(gen_test, output_signature=(tf.TensorSpec(shape=(32, 32, 3)), tf.TensorSpec(shape=(1,))))


    return train_set, test_set


def grad_normal(model, x, targets, train=False):
    with tf.GradientTape() as tape:
        y_pre = model(x, training=train)
        loss_value = LOSS_OBJECT(y_true=targets, y_pred=y_pre)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def plot_curve(train_acc, val_acc, train_loss, val_loss, path):

    plt.figure()
    plt.plot(train_acc, label='accuracy')
    plt.plot(val_acc, label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(path + "/acc.png")
    plt.close()

    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig(path + "/loss.png")
    plt.close()


def main():
    train_set, test_set = prepare_dataset()

    # model = ResNet50((224, 224, 3), 10)
    act_B = {"init_conv": 3.0, "init_depth_conv": 3.0, "linear": 3.0, "SE_1": [3.0] * 48, "SE_2": [3.0] * 48}
    model = PokeBNN(num_classes=10, phase=2, act_B=act_B).build(input_shape=(224, 224, 3))
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []

    result_path = "./train_result/PokeBNN_cifar10_adam_lr{}decay{}_epoch{}_bs{}_resize_regular_train".format(LR, DECAY_STEP, TOTAL_EPOCH, BATCH_SIZE)
    print(result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for epoch in range(TOTAL_EPOCH):
        epoch_train_loss_avg = tf.keras.metrics.Mean()
        epoch_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_test_loss_avg = tf.keras.metrics.Mean()
        epoch_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        train_set_train = train_set.apply(dataset_fn_train)
        test_set_train = test_set.apply(dataset_fn_test)

        for x, y in tqdm(train_set_train):
            # Optimize the model
            loss_value, grads = grad_normal(model, x, y, train=True)
            OPTIMIZER.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_train_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_train_accuracy.update_state(y, model(x, training=True))

        for x, y in tqdm(test_set_train):
            loss_value = LOSS_OBJECT(y_true=y, y_pred=model(x, training=False))

            # Track progress
            epoch_test_loss_avg.update_state(loss_value)  # Add current batch loss
            epoch_test_accuracy.update_state(y, model(x, training=False))

        # End epoch
        train_loss_results.append(epoch_train_loss_avg.result())
        train_accuracy_results.append(epoch_train_accuracy.result())
        test_loss_results.append(epoch_test_loss_avg.result())
        test_accuracy_results.append(epoch_test_accuracy.result())

        print("Epoch {:03d}: Train Loss: {:.3f}, Train Accuracy: {:.3%}, "
              "Test Loss: {:.3f}, Test Accuracy: {:.3%}".format(epoch, epoch_train_loss_avg.result(), epoch_train_accuracy.result(),
                                                                epoch_test_loss_avg.result(), epoch_test_accuracy.result()))

        plot_curve(train_accuracy_results, test_accuracy_results, train_loss_results, test_loss_results, result_path)
        if epoch % 20 == 0 or epoch == TOTAL_EPOCH-1:
            model.save_weights(result_path + "/model_{}.h5".format(epoch))


if __name__ == "__main__":
    main()
