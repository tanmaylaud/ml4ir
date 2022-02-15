import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.features.feature_fns.categorical import get_vocabulary_info
from ml4ir.applications.ranking.config.keys import PositionalBiasHandler
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.architectures.fixed_additive_positional_bias import FixedAdditivePositionalBias


OOV = 1


class DNNLayerKey:
    MODEL_NAME = "dnn"
    DENSE = "dense"
    BATCH_NORMALIZATION = "batch_norm"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    POSITIONAL_BIAS_HANDLER = "positional_bias_handler"
    CONCATENATED_INPUT = "concatenated_input"


class DNN(keras.Model):
    """Dense Neural Network architecture layer that maps features -> logits"""

    def __init__(self,
                 model_config: dict,
                 feature_config: FeatureConfig,
                 file_io: FileIO,
                 **kwargs):
        """
        Initialize a dense neural network layer

        Parameters
        ----------
        model_config: dict
            Dictionary defining the dense architecture spec
        feature_config: FeatureConfig
            FeatureConfig defining how each input feature is used in the model
        file_io: FileIO
            File input output handler
        """
        super().__init__(**kwargs)

        self.file_io: FileIO = file_io
        self.model_config = model_config
        self.feature_config = feature_config

        # Sort the train features dictionary so that we control the order
        # Concat all train features to get a dense feature vector
        self.train_features_op = layers.Concatenate(axis=-1, name=DNNLayerKey.CONCATENATED_INPUT)

        self.layer_ops: List = self.define_architecture(model_config, feature_config)
        if DNNLayerKey.POSITIONAL_BIAS_HANDLER in self.model_config and self.model_config[DNNLayerKey.POSITIONAL_BIAS_HANDLER][
                "key"] == PositionalBiasHandler.FIXED_ADDITIVE_POSITIONAL_BIAS:
            self.positional_bias_layer = FixedAdditivePositionalBias(max_ranks=self.model_config[DNNLayerKey.POSITIONAL_BIAS_HANDLER]["max_ranks"],
                                                                     kernel_initializer=self.model_config[DNNLayerKey.POSITIONAL_BIAS_HANDLER].get(
                                                                         "kernel_initializer", "Zeros"),
                                                                     l1_coeff=self.model_config[DNNLayerKey.POSITIONAL_BIAS_HANDLER].get(
                                                                         "l1_coeff", 0),
                                                                     l2_coeff=self.model_config[DNNLayerKey.POSITIONAL_BIAS_HANDLER].get("l2_coeff", 0))

    def define_architecture(self, model_config: dict, feature_config: FeatureConfig):
        """
        Convert the model from model_config to a List of tensorflow.keras.layer

        :param model_config: dict corresponding to the model config
        :param feature_config: dict corresponding to the feature config, only used in case of classification if the last
            layer of the model_config doesn't have a units number defined (or set to -1). In which case we retrieve the
            label vocabulary defined in the feature_config to deduce the number of units.
        :return: List[layers]: list of keras layer corresponding to each of the layers defined in the model_config.
        """
        def get_op(layer_type, layer_args):
            if layer_type == DNNLayerKey.DENSE:
                if not "units" in layer_args or layer_args["units"] == -1:
                    try:
                        label_feature_info = feature_config.get_label()
                        vocabulary_keys, vocabulary_ids = get_vocabulary_info(
                            label_feature_info["feature_layer_info"]["args"], self.file_io)
                        layer_args["units"] = len(vocabulary_keys) + OOV
                    except:
                        raise KeyError("We were not able to find information for the output layer of your DNN. "
                                       "Try specifying the number of output units either by passing \"units\" in the "
                                       "model configuration yaml file or units in the feature configuration file.")
                return layers.Dense(**layer_args)
            elif layer_type == DNNLayerKey.BATCH_NORMALIZATION:
                return layers.BatchNormalization(**layer_args)
            elif layer_type == DNNLayerKey.DROPOUT:
                return layers.Dropout(**layer_args)
            elif layer_type == DNNLayerKey.ACTIVATION:
                return layers.Activation(**layer_args)
            else:
                raise KeyError("Layer type is not supported : {}".format(layer_type))

        return [
            get_op(layer_args["type"], {k: v for k, v in layer_args.items() if k not in "type"})
            for layer_args in model_config["layers"]
        ]

    def build(self, input_shape):
        """Build the DNN model"""
        self.train_features = sorted(input_shape[FeatureTypeKey.TRAIN])

    def call(self, inputs, training=None):
        """
        Perform the forward pass for the architecture layer

        Parameters
        ----------
        inputs: dict of dict of tensors
            Input feature tensors divided as train and metadata
        training: bool
            Boolean to indicate if the layer is used in training or inference mode

        Returns
        -------
        tf.Tensor
            Logits tensor computed with the forward pass of the architecture layer
        """
        train_features = inputs[FeatureTypeKey.TRAIN]

        # Sort the train features dictionary so that we control the order
        # Concat all train features to get a dense feature vector
        layer_input = self.train_features_op([train_features[k] for k in sorted(train_features)])

        # Pass ranking features through all the layers of the DNN
        for layer_op in self.layer_ops:
            layer_input = layer_op(layer_input, training=training)

        if DNNLayerKey.POSITIONAL_BIAS_HANDLER in self.model_config and \
                self.model_config[DNNLayerKey.POSITIONAL_BIAS_HANDLER]["key"] == \
                PositionalBiasHandler.FIXED_ADDITIVE_POSITIONAL_BIAS:

            positional_bias = self.positional_bias_layer(
                metadata_features[self.feature_config.get_rank()["node_name"]],
                training=training)
            layer_input = tf.math.add(positional_bias, layer_input)

        # Collapse extra dimensions
        if isinstance(self.layer_ops[-1], layers.Dense) and (self.layer_ops[-1].units == 1):
            scores = tf.squeeze(layer_input, axis=-1)
        else:
            scores = layer_input

        return scores
