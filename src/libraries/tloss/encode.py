# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import os
import numpy
import src.libraries.tloss.scikit_wrappers as scikit_wrappers
from src.libraries.tloss.hyperparameters import Hyperparameters
from src.libraries.utils import channel_normalize
import gc
from torch import cuda


def fit_hyperparameters(X_train, train_labels, params: Hyperparameters, save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param X_train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    # Check the number of input channels
    params.in_channels = numpy.shape(X_train)[1]
    
    classifier.set_params(params)
    return classifier.fit(
        X_train, train_labels, save_memory=save_memory, verbose=True
    )

def encode(X_train, y_train, X_test, **kwargs):
    params = Hyperparameters(**kwargs)

    # TODO: Normalize data
    X_train, X_test = channel_normalize(X_train, X_test, channel_axis=1)

    # Flush cuda memory first
    gc.collect()
    cuda.empty_cache()

    # Train the model
    classifier = fit_hyperparameters(
            X_train, y_train, params=params, save_memory=False
        )

    # Embed
    rep_train = classifier.encode(X_train)
    rep_test = classifier.encode(X_test)

    return rep_train, rep_test