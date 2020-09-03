import os
import tempfile
import pytest

import keras
from keras import layers
import numpy as np

from combine_layer import Combine

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Normalise the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Layers:
#[Tops, 'Trouser', Shoes, 'Bag']
#['T-shirt/top', 'Pullover', 'Dress', 'Coat', 'Shirt']
#['Sandal', 'Sneaker', 'Ankle boot']

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

# Test Normal Softmax vs. Combine Layer 
def test_model_baseline():
    model_baseline = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(10, activation='softmax')
        ])

    model_baseline.compile(optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    print('Testing baseline softmax:\n')
    model_baseline.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model_baseline.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

def test_hrch_model():
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

    print('\nTesting combine layer:\n')
    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

#Test Reduction Options
def test_reduction_opts():
    reduction_opts = ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

    for reduction in reduction_opts:
        print('\nTesting reduction with %s:\n' % reduction)
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            Combine([
                layers.Dense(4, activation='softmax', name='root'),
                layers.Dense(5, activation='softmax', name='tops'),
                layers.Dense(3, activation='softmax', name='shoes')],
                output_spec = output_spec,
                reduction=reduction)
            ])

        model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=1)

        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)


# Test too long output_spec
def test_long_output_spec():
    print('\nTesting too long output_spec elements:\n')
    with pytest.raises(AssertionError):
        output_spec_invalid = [
                [0, 0, 0, 0], #T-shirt/top
                [1],
                [0, 1],
                [0, 2], #Dress
                [0, 3],
                [2, -1, 0], # Sandal
                [0, 4],
                [2, -1, 1],
                [3],
                [2, -1, 2] # Ankle boot
                ]

        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            Combine([
                layers.Dense(4, activation='softmax', name='root'),
                layers.Dense(5, activation='softmax', name='tops'),
                layers.Dense(3, activation='softmax', name='shoes')],
                output_spec = output_spec_invalid)
            ])

        model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=1)
        print('Completed model fit from too long output_spec (wrong)')

# Test invalid reduction option
def test_invalid_reduction():
    print('\nTesting invalid reduction option:\n')
    with pytest.raises(AssertionError):
        Combine([
            layers.Dense(4, activation='softmax', name='root'),
            layers.Dense(5, activation='softmax', name='tops'),
            layers.Dense(3, activation='softmax', name='shoes')],
            output_spec = output_spec,
            reduction='xyz')
        print('Layer built with invalid reduction option (wrong)')

# Test serialization and deserialization
def test_serialization_deserialization():
    print('\nTesting serialization and deserialization:\n')

    with tempfile.TemporaryDirectory() as save_dir:
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

        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy before serialization:', test_acc)

        pred = model.predict(test_images)

        # saving the weights worked with tensorflow 1.15.2 but not 2.1
        model.save_weights(os.path.join(save_dir, 'model'))
        model.load_weights(os.path.join(save_dir, 'model'))

        print('\nPredictions same after reloading weights:', (pred == model.predict(test_images)).all())
        assert (pred == model.predict(test_images)).all()

        model_path = os.path.join(save_dir, 'model.h5')
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'Combine': Combine})

        print('Predictions same after reloading entire model:', (pred == model.predict(test_images)).all())
        assert (pred == model.predict(test_images)).all()
