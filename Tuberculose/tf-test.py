# from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10)
# ])

# predictions = model(x_train[:1]).numpy()
# print("predictions", predictions)
# print("tf", tf.nn.softmax(predictions).numpy())
# print(x_train[0])

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# loss_fn(y_train[:1], predictions).numpy()

# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test,  y_test, verbose=2)

# probability_model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.Softmax()
# ])

# probability_model(x_test[:5])


# mammal = tf.Variable("Elephant", tf.string)
# ignition = tf.Variable(451, tf.int16)
# floating = tf.Variable(3.14159265359, tf.float64)
# its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

# print(mammal)
# print(ignition)
# print(floating)
# print(its_complicated)

# mystr = tf.Variable(["Hello"], tf.string)
# cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
# first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
# its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# print(mystr)
# print(cool_numbers)
# print(first_primes)
# print(its_very_complicated)

from tensorflow.python.platform import build_info as tf_build_info
print(tf_build_info.cuda_version_number)
# 9.0 in v1.10.0
print(tf_build_info.cudnn_version_number)
# 7 in v1.10.0
