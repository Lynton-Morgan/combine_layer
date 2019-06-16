import keras
import keras.backend as K

class Combine(keras.layers.Layer):
    """Combine layer

    This layer recombines the output of its internal layers
    
    #Arguments
    layers: A list of Keras layers
    output_spec: A list of integer lists, indices from each layer in 'layers'
        that make up each output coordinate
    reduction: A string, the function to use between layer coordinates

    #Example
    To make a 3-element softmax binary tree:

    output_spec = [[0, 0], [0, 1], [1, -1]]
    comb = Combine([Dense(2, activation='softmax'),
                    Dense(2, activation='softmax')],
                    output_spec=output_spec,
                    reduction='prod')
    """
    def __init__(self, layers, output_spec, reduction='prod', **kwargs):
        self.layers = layers
        assert len(layers) > 0, "Must have layers in 'layers'"

        self.output_spec = output_spec
        for idx_spec in output_spec:
            assert len(idx_spec) <= len(layers), \
            "Length of each element in output_spec must not exceed the number of layers"

        self.output_dim = len(output_spec)

        self.reduction = reduction
        reducer_dict = {
                'max': K.max,
                'mean': K.mean,
                'min': K.min,
                'prod': K.prod,
                'std': K.std,
                'sum': K.sum,
                'var': K.var,
                }
        assert reduction in reducer_dict, "'reduction' must be one of %s" % (list(reducer_dict.keys()))
        self.reducer = reducer_dict[reduction]

        super(Combine, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []

        for layer in self.layers:
            layer.build(input_shape)
            self.trainable_weights += layer.trainable_weights

        super(Combine, self).build(input_shape)

    def call(self, inputs):
        layer_outputs = [layer(inputs) for layer in self.layers]

        outputs = []
        for indices in self.output_spec:
            var = K.stack(
                    [layer_outputs[layer_idx][...,idx] for layer_idx, idx in enumerate(indices) if idx >= 0],
                    axis=0)
            outputs.append(self.reducer(var, axis=0))

        result = K.stack(outputs, axis=-1)
        return result

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

