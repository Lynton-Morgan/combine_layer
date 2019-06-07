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
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation='softmax')
    ])
```

To make a hierarchical softmax with a Combine layer, we pass the component softmax layers and an output specification to the class constructor.

The way in which the layers are combined is defined by `output_spec`. `output_spec` is a list of lists; the length of the outer list is the number of output elements, in this example it is 10. Each inner list contains indices from the component layers, up to one from each. `-1` at position `i` indicates that layer `i` is not used.
```python
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
```

Each output element is the product of its components by default, but other reduction options are implemented.
