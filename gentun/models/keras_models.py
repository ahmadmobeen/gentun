#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras
"""

import keras.backend as K

from keras.layers import Input, Conv2D, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import KFold

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


class GeneticCnnModel(GentunModel):

    def __init__(self, x_train, y_train, genes, input_shape, kernels_per_layer,
                 kernel_sizes, dense_units, dropout_probability, classes,
                 nfold=5, epochs=3, batch_size=128, plot_model=False):
        super(GeneticCnnModel, self).__init__(x_train, y_train)
        self.model = self.build_model(
            genes, input_shape, kernels_per_layer, kernel_sizes, dense_units, dropout_probability, classes
        )
        self.nfold = nfold
        self.epochs = epochs
        self.batch_size = batch_size
        if plot_model:
            # Draw model to validate gene-to-DAG
            from keras.utils import plot_model
            plot_model(self.model, to_file='model.png')

    @staticmethod
    def build_dag(x, connections, kernels):
        # Get number of nodes (K_s) using the fact that K_s*(K_s-1)/2 == #bits
        nodes = int((1 + (1 + 8 * len(connections)) ** 0.5) / 2)
        # Separate bits by whose input they represent (GeneticCNN paper uses a dash)
        ctr = 0
        idx = 0
        separated_connections = []
        while idx + ctr < len(connections):
            ctr += 1
            separated_connections.append(connections[idx:idx + ctr])
            idx += ctr
        # Get outputs by node (dummy output ignored)
        outputs = []
        for node in range(nodes - 1):
            node_outputs = []
            for i, node_connections in enumerate(separated_connections[node:]):
                if node_connections[node] == '1':
                    node_outputs.append(node + i + 1)
            outputs.append(node_outputs)
        outputs.append([])
        # Get inputs by node (dummy input, x, ignored)
        inputs = [[]]
        for node in range(1, nodes):
            node_inputs = []
            for i, connection in enumerate(separated_connections[node - 1]):
                if connection == '1':
                    node_inputs.append(i)
            inputs.append(node_inputs)
        # Build DAG
        output_vars = []
        all_vars = [None] * nodes
        for i, (ins, outs) in enumerate(zip(inputs, outputs)):
            if ins or outs:
                if not ins:
                    tmp = x
                else:
                    add_vars = [all_vars[i] for i in ins]
                    if len(add_vars) > 1:
                        tmp = Add()(add_vars)
                    else:
                        tmp = add_vars[0]
                tmp = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(tmp)
                tmp = Activation('relu')(tmp)
                all_vars[i] = tmp
                if not outs:
                    output_vars.append(tmp)
        if len(output_vars) > 1:
            return Add()(output_vars)
        return output_vars[0]

    def build_model(self, genes, input_shape, kernels_per_layer, kernel_sizes, dense_units,
                    dropout_probability, classes):
        x_input = Input(input_shape)
        x = x_input
        for layer, kernels in enumerate(kernels_per_layer):
            # Default input node
            x = Conv2D(kernels, kernel_size=kernel_sizes[layer], strides=(1, 1), padding='same')(x)
            x = Activation('relu')(x)
            # Decode internal connections
            connections = genes['S_{}'.format(layer + 1)]
            # If at least one bit is 1, then we need to construct the Directed Acyclic Graph
            if not all([not bool(int(connection)) for connection in connections]):
                x = self.build_dag(x, connections, kernels)
                # Output node
                x = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_probability)(x)
        x = Dense(classes, activation='softmax')(x)
        return Model(inputs=x_input, outputs=x, name='GeneticCNN')

    def reset_weights(self):
        """Initialize model weights."""
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def cross_validate(self):
        """Train model using k-fold cross validation and
        return mean value of the loss.
        """
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        loss = .0
        kfold = KFold(n_splits=self.nfold, shuffle=True)  # TODO: implement stratified k-fold
        for fold, (train, test) in enumerate(kfold.split(self.x_train)):
            print("KFold {}/{}".format(fold + 1, self.nfold))
            self.reset_weights()
            self.model.fit(
                self.x_train[train], self.y_train[train], epochs=self.epochs, batch_size=self.batch_size, verbose=1
            )
            loss += self.model.evaluate(self.x_train[test], self.y_train[test], verbose=0) / self.nfold
        return loss
