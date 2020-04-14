# Combine Layer

The Combine layer is a custom Keras layer that contains other Keras layers and recombines their output. Its component layers all take the same input.

The original goal behind this was to make a hierarchical softmax layer, but this layer is flexible and can be made in many other ways.

## Example

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset has 10 classes which are different items of clothing.

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

In the Keras API, the typical way to target these classes is with a softmax in the output layer, e.g.
```python
import keras
from keras import layers

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Normalise the data
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam', 
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)
```

To make a hierarchical softmax with a Combine layer, we pass the component softmax layers and an output specification to the class constructor.

The way in which the layers are combined is defined by `output_spec`. `output_spec` is a list of lists; the length of the outer list is the number of output elements, in this example it is 10. Each inner list contains indices from the component layers, up to one from each. `-1` at position `i` indicates that layer `i` is not used.
```python
from combine_layer import Combine

# Root Layer: Size 4 [Tops, Trouser, Shoes, Bag]
# Tops Layer: Size 5 [T-shirt/top, Pullover, Dress, Coat, Shirt]
# Shoes Layer: Size 3 [Sandal, Sneaker, Ankle boot]

output_spec = [
        [0, 0],    # T-shirt/top
        [1],       # Trouser
        [0, 1],    # Pullover
        [0, 2],    # Dress
        [0, 3],    # Coat
        [2, -1, 0],# Sandal
        [0, 4],    # Shirt
        [2, -1, 1],# Sneaker
        [3],       # Bag
        [2, -1, 2] # Ankle boot
        ]

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    Combine([
        layers.Dense(4, activation='softmax', name='root'),
        layers.Dense(5, activation='softmax', name='tops'),
        layers.Dense(3, activation='softmax', name='shoes')],
        output_spec = output_spec)
    ])

model.compile(optimizer='adam', 
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)
```

Each output element is the product of its components by default, but other reduction options are implemented.

## Saving and Loading

The weights of a Keras model containing a Combine layer can be saved and loaded as follows:
```python
import os
import keras
from keras import layers

from combine_layer import Combine

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    Combine([
        layers.Dense(4, activation='softmax', name='root'),
        layers.Dense(5, activation='softmax', name='tops'),
        layers.Dense(3, activation='softmax', name='shoes')],
        output_spec = output_spec)
    ])

model.compile(optimizer='adam', 
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

model.save_weights(os.path.join(save_dir, 'model'))
model.load_weights(os.path.join(save_dir, 'model'))
```
This works when using TensorFlow 1.15 as a backend but not TensorFlow 2.1

To load an entire Keras model containing a Combine layer, the Combine layer has to be specified as a custom object

```python
from combine_layer import Combine

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    Combine([
        layers.Dense(4, activation='softmax', name='root'),
        layers.Dense(5, activation='softmax', name='tops'),
        layers.Dense(3, activation='softmax', name='shoes')],
        output_spec = output_spec)
    ])

model.compile(optimizer='adam', 
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

# Saving the entire model
model_path = os.path.join(save_dir, 'model.h5')
model.save(model_path)

# Loading the entire model from a file
model = keras.models.load_model(model_path, custom_objects={'Combine': Combine})
```

