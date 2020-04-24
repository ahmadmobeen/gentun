#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras
"""

import keras.backend as K
import numpy as np

from keras.layers import Input, Conv2D, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras import metrics
# from keras.utils import multi_gpu_model
import os

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


class GeneticCnnModel(GentunModel):

    def __init__(self, gpu,x_train, y_train, x_test,y_test,genes, nodes, input_shape, kernels_per_layer, kernel_sizes, dense_units,
                 dropout_probability, classes, kfold=5, epochs=(3,), learning_rate=(1e-3,), batch_size=32):
        super(GeneticCnnModel, self).__init__(x_train, y_train,x_test,y_test)
        self.gpu=gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        self.model = self.build_model(
            genes, nodes, input_shape, kernels_per_layer, kernel_sizes,
            dense_units, dropout_probability, classes
        )
        # support for multi-gpus
        # try:
        #     self.parallel_model = multi_gpu_model(self.model, gpus=2)
        #     print("Training using multiple GPUs..")
        # except ValueError:
        self.parallel_model = self.model
        print("Training using single GPU..")

        self.name = '-'.join(gene for gene in genes.values())
        self.kfold = kfold
        if type(epochs) is int and type(learning_rate) is float:
            self.epochs = epochs#(epochs,)
            self.learning_rate = learning_rate#(learning_rate,)
        elif type(epochs) is tuple and type(learning_rate) is tuple:
            self.epochs = epochs
            self.learning_rate = learning_rate
        else:
            print(epochs, learning_rate)
            raise ValueError("epochs and learning_rate must be both either integers or tuples of integers.")
        self.batch_size = batch_size

    def plot(self):
        """Draw model to validate gene-to-DAG."""
        from keras.utils import plot_model
        plot_model(self.model, to_file='{}.png'.format(self.name))

    @staticmethod
    def build_dag(x, nodes, connections, kernels):
        # Get number of nodes (K_s) using the fact that K_s*(K_s-1)/2 == #bits
        # nodes = int((1 + (1 + 8 * len(connections)) ** 0.5) / 2)
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

    def build_model(self, genes, nodes, input_shape, kernels_per_layer, kernel_sizes,
                    dense_units, dropout_probability, classes):
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
                x = self.build_dag(x, nodes[layer], connections, kernels)
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
        return mean value of the validation accuracy.
        """
        loss=.0
        acc = .0
        mae= .0
        mse=.0
        msle=.0
        kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        for fold, (train, validation) in enumerate(kfold.split(self.x_train, np.where(self.y_train == 1)[1])):
            print("KFold {}/{}".format(fold + 1, self.kfold))
            self.reset_weights()
            for epochs, learning_rate in zip(self.epochs, self.learning_rate):
                print("Training {} epochs with learning rate {}".format(epochs, learning_rate))
                self.parallel_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy',metrics.mae,metrics.mse,metrics.msle])
                self.parallel_model.fit(
                    self.x_train[train], self.y_train[train], epochs=epochs, batch_size=self.batch_size, verbose=1
                )
            results=self.parallel_model.evaluate(self.x_train[validation], self.y_train[validation], verbose=0)
            # print(results)
            loss += results[0] / self.kfold
            acc += results[1] / self.kfold
            mae += results[2] / self.kfold
            mse += results[3] / self.kfold
            msle += results[4] / self.kfold
        K.clear_session()
        return loss, acc, mae, mse, msle

    def old_validate(self):
        """Train model using k-fold cross validation and
        return mean value of the validation accuracy.
        """
        loss=.0
        acc = .0
        mae= .0
        mse=.0
        msle=.0


        self.reset_weights()
        for epochs, learning_rate in zip(self.epochs, self.learning_rate):
            print("Training {} epochs with learning rate {}".format(epochs, learning_rate))
            self.parallel_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy',metrics.mae,metrics.mse,metrics.msle])
            self.parallel_model.fit(
                self.x_train, self.y_train, epochs=epochs, batch_size=self.batch_size, verbose=1
            )
        results=self.parallel_model.evaluate(self.x_test, self.y_test, verbose=0)
        # print(results)
        loss = results[0]
        acc = results[1]
        mae = results[2]
        mse = results[3]
        msle = results[4]
        K.clear_session()
        return loss, acc, mae, mse, msle

    # def lr_decay(self,epoch):
    #     initial_lrate = 0.1
    #     k = 0.1
    #     lrate = initial_lrate * exp(-k * t)
    #     return lratelrate = LearningRateScheduler(exp_decay)

    def validate(self):
        """Train model using k-fold cross validation and
        return mean value of the validation accuracy.
        """
        early_stopper= EarlyStopping(monitor='val_acc',patience=30)

        self.reset_weights()
        print("Training {} epochs with learning rate {}".format(self.epochs, self.learning_rate))
        self.parallel_model.compile(optimizer=Adam(lr=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy',metrics.mae,metrics.mse,metrics.msle])
        training_results=self.parallel_model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1,callbacks=[early_stopper],validation_split=0.1)
        # print("Training Results",str(training_results.history))
        print ("Stopped after {} Epochs".format(len(training_results.epoch)))
        # print("Model",training_results.model.summary())

        results=self.parallel_model.evaluate(self.x_test, self.y_test, verbose=0)
        # print(results)
        loss = results[0]
        acc = results[1]
        mae = results[2]
        mse = results[3]
        msle = results[4]
        K.clear_session()
        return loss, acc, mae, mse, msle,training_results.history,training_results.epoch,training_results.model.to_json()


