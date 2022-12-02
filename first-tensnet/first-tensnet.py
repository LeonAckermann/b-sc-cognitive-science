import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

# 2.1 - loading mnist data set
(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

# inspect data

# 2.2 - set up data pipiline
def prepare_mnist_data(mnist):
    # map from uint8 to tf.float
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # flatten input
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1, )), target))
    
    # normalize input to gaussian distribution or divide by 128
    mnist = mnist.map(lambda img, target: (((img/128)-1), target))

    # encode labels as one hot vector
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # keep the progess in memory
    mnist = mnist.cache()
    mnist = mnist.shuffle(1000) 
    mnist = mnist.batch(32) # 32 image in one batch
    mnist = mnist.prefetch(20) # prepare 20 next datapoints 

    return mnist

train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)


# 2.3 - build network
class MyModel(tf.keras.Model):
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x
        


# 2.4 - training the network
def train_step(model, input, target, loss_function, optimizer):

    # loss object and optimizer and are instances of respective tensorflow classes 
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables) # all variables with trainable = True
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # updating weights with optimizer
    return loss

# 2.4.1 - testing the model
def test(model, test_data, loss_function):

    test_accuracy_aggregator = []
    test_loss_aggregator = [] # continuous

    # input is batch of 32 examples
    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


# Training
epochs = 10
learning_rate = 0.001

model = MyModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

# for visualization
train_losses = []
test_losses = []
test_accuracies = []

# test model before training
test_loss, test_accuracy = test(model, test_dataset)



# 2.5 - visualization
def visualization ( train_losses , train_accuracies , test_losses , test_accuracies ):
    """
    Visualizes accuracy and loss for training and test data using the mean of each epoch. 
    Loss is displayed in a regular line , accuracy in a dotted line. 
    Training data is displayed in blue , test data in red .

    Parameters
    ----------
    train_losses : numpy . ndarray
    training losses
    train_accuracies : numpy . ndarray
    training accuracies
    test_losses : numpy . ndarray
    test losses
    test_accuracies : numpy . ndarray
    test accuracies
    """


    plt . figure ()
    line1 , = plt. plot ( train_losses , "b-")
    line2 , = plt. plot ( test_losses , "r-")
    line3 , = plt. plot ( train_accuracies , "b:")
    line4 , = plt. plot ( test_accuracies , "r:")
    plt . xlabel (" Training steps ")
    plt . ylabel (" Loss / Accuracy ")
    plt . legend (( line1 , line2 , line3 , line4 ), (" training loss ", " testloss ", " train accuracy ", " test accuracy "))
    plt . show ()

