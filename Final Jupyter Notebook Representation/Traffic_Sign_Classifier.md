
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

# Route to data
DATA_ROUTE = '../traffic-classifier-data/'

training_file = DATA_ROUTE + 'train.p'
validation_file= DATA_ROUTE + 'valid.p'
testing_file = DATA_ROUTE + 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

import numpy as np

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print('Train shape', X_train.shape)
print("Number of training examples =", n_train)
print("Number of valid examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Train shape (34799, 32, 32, 3)
    Number of training examples = 34799
    Number of valid examples = 4410
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
import csv
# Visualizations will be shown in the notebook.
%matplotlib inline

# get names from csv
values = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('code','i8'), ('name','U55')], delimiter=',')
# show image of 10 random data points
fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(10):
    index = random.randint(0, len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(values[y_train[index]][1])
```


![png](output_8_0.png)



```python

train_ids, train_counts = np.unique(y_train, return_counts=True)
plt.bar(train_ids, train_counts)
plt.grid()
plt.title("Training Dataset Id Counts")
plt.show()

test_ids, test_counts = np.unique(y_test, return_counts=True)
plt.bar(test_ids, test_counts)
plt.grid()
plt.title("Testing Dataset Id Counts")
plt.show()

valid_ids, valid_counts = np.unique(y_valid, return_counts=True)
plt.bar(valid_ids, valid_counts)
plt.grid()
plt.title("Validation Dataset Id Counts")
plt.show()
```


![png](output_9_0.png)



![png](output_9_1.png)



![png](output_9_2.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
X_train_rgb = X_train
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)

X_valid_rgb = X_valid
X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)

print('Train RGB', X_train_rgb.shape)
print('Train gray scale', X_train_gray.shape)

print('Valid RGB', X_valid_rgb.shape)
print('Valid gray scale', X_valid_gray.shape)

print('Test RGB', X_test_rgb.shape)
print('Test gray scale', X_test_gray.shape)
```

    Train RGB (34799, 32, 32, 3)
    Train gray scale (34799, 32, 32, 1)
    Valid RGB (4410, 32, 32, 3)
    Valid gray scale (4410, 32, 32, 1)
    Test RGB (12630, 32, 32, 3)
    Test gray scale (12630, 32, 32, 1)


## Visualize rgb vs grayscale


```python
def print_rgb_vs_gray(n_rows, n_cols, offset, rgb_images, gray_images):
    fig, axs = plt.subplots(n_rows,n_cols, figsize=(18, 14))
    fig.subplots_adjust(hspace = .1, wspace=.001)
    axs = axs.ravel()
    for j in range(0,n_rows,2):
        for i in range(n_cols):
            index = i + j*n_cols
            image = rgb_images[index + offset]
            axs[index].axis('off')
            axs[index].imshow(image)
        for i in range(n_cols):
            index = i + j*n_cols + n_cols 
            image = gray_images[index + offset - n_cols].squeeze()
            axs[index].axis('off')
            axs[index].imshow(image, cmap='gray')
```


```python
print('Training data')
print_rgb_vs_gray(4, 4, 100, X_train_rgb, X_train_gray)
```

    Training data



![png](output_16_1.png)



```python
print('Validation data')
print_rgb_vs_gray(4, 4, 100, X_valid_rgb, X_valid_gray)
```

    Validation data



![png](output_17_1.png)



```python
print('Test data')
print_rgb_vs_gray(4, 4, 100, X_test_rgb, X_test_gray)
```

    Test data



![png](output_18_1.png)



```python
### Normalizing values from -1 to 1
train_mean = np.mean(X_train_gray)
train_std = np.std(X_train_gray)
X_train_normalized = (X_train_gray - train_mean)/train_std

test_mean = np.mean(X_test_gray)
test_std = np.std(X_test_gray)
X_test_normalized = (X_test_gray - test_mean)/test_std

valid_mean = np.mean(X_valid_gray)
valid_std = np.std(X_valid_gray)
X_valid_normalized = (X_valid_gray - valid_mean)/valid_std

# ## The mean is negligibly close zero
print(np.mean(X_train_normalized))
print(np.mean(X_test_normalized))
print(np.mean(X_valid_normalized))
```

    8.21244085217e-16
    1.58103494227e-15
    3.86689924223e-17



```python
def print_gray_vs_normalized(index, original, normalized):
    fig, axs = plt.subplots(1,2, figsize=(10, 3))
    axs = axs.ravel()
    axs[0].axis('off')
    axs[0].set_title('original')
    axs[0].imshow(original[index].squeeze(), cmap='gray')

    axs[1].axis('off')
    axs[1].set_title('normalized')
    axs[1].imshow(normalized[index].squeeze(), cmap='gray')
```


```python
print('Train Gray vs Normalized')
print_gray_vs_normalized(0, X_train_gray, X_train_normalized)
```

    Train Gray vs Normalized



![png](output_21_1.png)



```python
print('Valid Gray vs Normalized')
print_gray_vs_normalized(0, X_valid_gray, X_valid_normalized)
```

    Valid Gray vs Normalized



![png](output_22_1.png)



```python
print('Test Gray vs Normalized')
print_gray_vs_normalized(0, X_test_gray, X_test_normalized)
```

    Test Gray vs Normalized



![png](output_23_1.png)


### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

mu = 0
sigma = 0.1

def Convolution(x, a, depth):
    conv_W = tf.Variable(tf.truncated_normal(shape=(5,5,a,depth), mean = mu, stddev = sigma))
    conv_B = tf.Variable(tf.zeros(depth))
    conv = tf.nn.conv2d(x, conv_W, strides=[1,1,1,1], padding='VALID') + conv_B
    return tf.nn.relu(conv)

def Pooling(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

def FullyConnected(x, a, c):
    W = tf.Variable(tf.truncated_normal(shape=(a,c), mean = mu, stddev = sigma))
    B = tf.Variable(tf.zeros(c))
    return tf.matmul(x, W) + B

def LeNet(x):    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    x = Convolution(x, 1, 6)
    
    # Pooling: Input 28x28x6. Output = 14x14x6
    x = Pooling(x)
    
    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    x = Convolution(x, 6, 16)
    
    #Pooling: Input = 10x10x16. Output = 5x5x16
    x = Pooling(x)
    
    # Flatten. Input = 5x5x16. Output = 400.
    x = flatten(x)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    x = FullyConnected(x, 400, 120)
    
    # Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    x = FullyConnected(x, 120, 84)
    
    # Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)
    
    # Layer 5: Fully Connected, Input 84. Output = 43.
    return FullyConnected(x, 84, 43)
```


```python
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

### Train Pipeline


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# Learning rate
rate = 0.0009
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

## Model Evaluation
Evaluate how well the loss and accuracy of the model for a given dataset.
You do not need to modify this section.


```python
EPOCHS = 25
BATCH_SIZE = 100


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

## Train the Model

Run the training data through the training pipeline to train the model.
Before each epoch, shuffle the training set.
After each epoch, measure the loss and accuracy of the validation set.
Save the model after training.
You do not need to modify this section.


```python
from sklearn.utils import shuffle
```


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_normalized)
    
    print("Training...")
    print()
    X_training = X_train_normalized
    Y_training = y_train
    for i in range(EPOCHS):
        X_training, Y_training = shuffle(X_training, Y_training)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_training[offset:end], Y_training[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_valid_normalized, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

    print()
    training_accuracy = evaluate(X_train_normalized, y_train)
    print("Train Accuracy = {:.3f}".format(training_accuracy))
    final_validation_accuracy = evaluate(X_valid_normalized, y_valid)
    print("Validation Accuracy = {:.3f}".format(final_validation_accuracy))
    test_accuracy = evaluate(X_test_normalized, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    saver.save(sess, 'lenet')
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.603
    EPOCH 2 ...
    Validation Accuracy = 0.800
    EPOCH 3 ...
    Validation Accuracy = 0.855
    EPOCH 4 ...
    Validation Accuracy = 0.876
    EPOCH 5 ...
    Validation Accuracy = 0.901
    EPOCH 6 ...
    Validation Accuracy = 0.905
    EPOCH 7 ...
    Validation Accuracy = 0.928
    EPOCH 8 ...
    Validation Accuracy = 0.921
    EPOCH 9 ...
    Validation Accuracy = 0.931
    EPOCH 10 ...
    Validation Accuracy = 0.937
    EPOCH 11 ...
    Validation Accuracy = 0.939
    EPOCH 12 ...
    Validation Accuracy = 0.937
    EPOCH 13 ...
    Validation Accuracy = 0.944
    EPOCH 14 ...
    Validation Accuracy = 0.945
    EPOCH 15 ...
    Validation Accuracy = 0.953
    EPOCH 16 ...
    Validation Accuracy = 0.948
    EPOCH 17 ...
    Validation Accuracy = 0.939
    EPOCH 18 ...
    Validation Accuracy = 0.948
    EPOCH 19 ...
    Validation Accuracy = 0.953
    EPOCH 20 ...
    Validation Accuracy = 0.950
    EPOCH 21 ...
    Validation Accuracy = 0.950
    EPOCH 22 ...
    Validation Accuracy = 0.952
    EPOCH 23 ...
    Validation Accuracy = 0.955
    EPOCH 24 ...
    Validation Accuracy = 0.955
    EPOCH 25 ...
    Validation Accuracy = 0.950
    
    Train Accuracy = 0.996
    Validation Accuracy = 0.950
    Test Accuracy = 0.937


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
# load OpenCV
import cv2

#reading in an image
import glob
```


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

fig, axs = plt.subplots(2,3, figsize=(4, 2))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

my_images = []

for i in range(0, 6):
    image = cv2.cvtColor(cv2.imread('./images/'+str(i+1)+'.png'), cv2.COLOR_BGR2RGB)
    axs[i].axis('off')
    axs[i].imshow(image)
    my_images.append(image)

my_images_np = np.array(my_images)
my_images_gry = np.sum(my_images_np/3, axis=3, keepdims=True)

new_mean = np.mean(my_images_gry)
new_std = np.std(my_images_gry)
my_images_normalize = (my_images_gry - new_mean)/new_std

# Images' Ids
new_images_labels = np.array([1, 22, 35, 15, 37, 18])
```


![png](output_39_0.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #lenet = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    total_accuracy = 0
    for i in range(6):
        img_accuracy = evaluate([my_images_normalize[i]], [new_images_labels[i]])
        total_accuracy += img_accuracy
        print('Image {}'.format(i+1))
        print("Accuracy = {:.3f}".format(img_accuracy))
        print()
    print("Accuracy for 6 images = {:.3f}".format(total_accuracy/6))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Image 1
    Accuracy = 1.000
    
    Image 2
    Accuracy = 1.000
    
    Image 3
    Accuracy = 1.000
    
    Image 4
    Accuracy = 1.000
    
    Image 5
    Accuracy = 1.000
    
    Image 6
    Accuracy = 1.000
    
    Accuracy for 6 images = 1.000


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
```

## Total accuracy shown for the 6 images is 100%

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
k_size = 5
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=k_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalize, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalize, keep_prob: 1.0})
    for i in range(6):
        print('For image {}'.format(new_images_labels[i]))
        for j in range(k_size):
            print('Guess {} : ({:.3f})'.format(j+1, my_top_k[0][i][j]))
        print()
```

    INFO:tensorflow:Restoring parameters from ./lenet
    For image 1
    Guess 1 : (1.000)
    Guess 2 : (0.000)
    Guess 3 : (0.000)
    Guess 4 : (0.000)
    Guess 5 : (0.000)
    
    For image 22
    Guess 1 : (0.980)
    Guess 2 : (0.020)
    Guess 3 : (0.001)
    Guess 4 : (0.000)
    Guess 5 : (0.000)
    
    For image 35
    Guess 1 : (1.000)
    Guess 2 : (0.000)
    Guess 3 : (0.000)
    Guess 4 : (0.000)
    Guess 5 : (0.000)
    
    For image 15
    Guess 1 : (0.481)
    Guess 2 : (0.256)
    Guess 3 : (0.092)
    Guess 4 : (0.073)
    Guess 5 : (0.051)
    
    For image 37
    Guess 1 : (1.000)
    Guess 2 : (0.000)
    Guess 3 : (0.000)
    Guess 4 : (0.000)
    Guess 5 : (0.000)
    
    For image 18
    Guess 1 : (1.000)
    Guess 2 : (0.000)
    Guess 3 : (0.000)
    Guess 4 : (0.000)
    Guess 5 : (0.000)
    


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```
