# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        activation_fn = tf.pow(vector,3)
        return activation_fn
        # Comment the next line after implementing call.
        # raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        import pdb
        # pdb.set_trace()

        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.num_tokens=num_tokens
        self.hidden_dim=hidden_dim
        self.num_transitions=num_transitions
        self.trainable_embeddings=trainable_embeddings

        #Initialising Embeddings with size (vocab_size,embedding_dim). Setting it as trainable=true only during Training phase.
        self.embeddings = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_dim],
                                                                 mean=0, stddev=0.01),trainable=self.trainable_embeddings)

        #Initialising Layer Weights and Bias for hidden layer (using trunctated_normal):
        self.weights1 = tf.Variable(
            tf.random.truncated_normal(shape=[self.hidden_dim, self.embedding_dim * self.num_tokens],
                                       mean=0, stddev=tf.sqrt(2/(self.hidden_dim+(self.embedding_dim * self.num_tokens)))), trainable=True)

        self.bias = tf.Variable(tf.random.truncated_normal(shape=[1, self.hidden_dim],
                                                           mean=0, stddev=tf.sqrt(2/self.hidden_dim)), trainable=True)

        #Initialising Weights for Output layer:
        self.weights2 = tf.Variable(tf.random.truncated_normal(shape=[self.num_transitions, self.hidden_dim],
                                                               mean=0, stddev=tf.sqrt(2/(self.num_transitions + self.hidden_dim))),trainable=True)


        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        # import pdb;
        # pdb.set_trace()
        self.tokens = tf.nn.embedding_lookup(self.embeddings, inputs)
        ##Defining Hidden Layer:
        batch_size=self.tokens.shape[0]
        reshaped_tokens = tf.reshape(self.tokens,[batch_size,-1])
        hidden_layer = self._activation(tf.matmul(reshaped_tokens,tf.transpose(self.weights1))
                                        + tf.matmul(tf.ones([batch_size,1]),self.bias))
        logits = tf.matmul(hidden_layer,tf.transpose(self.weights2))

        # TODO(Students) End

        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        label_mask = tf.cast(labels>=0, tf.float32)     #Creating mask to filter out infeasible transitions [-1].
        feasible_logits = tf.math.multiply(logits, label_mask)  # Retrieving only logits with feasible transitions [0,1]
        exp_feasible_logits = tf.math.exp(feasible_logits)      # Computing numerator e^(feasible_logits) for softmax.
        masked_exp_feasible_logits = tf.math.multiply(label_mask, exp_feasible_logits)  # Retrieving exp. logits for only Feasible transitions. Masking e^[infeasible_transitions] with 0
        sum_masked_exp_feasible_logits = tf.matmul(tf.math.reduce_sum(masked_exp_feasible_logits, keepdims=True, axis=1),
                                                   tf.ones([1, self.num_transitions]))  # Computing Denominator term for softmax.
        softmax_logits = tf.math.divide(masked_exp_feasible_logits, sum_masked_exp_feasible_logits) # Final Softmax Logits.

        ##Finding Cross Entropy Loss as per Manning paper:
        log_prob = tf.math.log(softmax_logits + 1e-10)      # adding small exponent value to avoid NaN's
        loss_mask = tf.cast(labels==1,tf.float32)
        correct_logits=tf.math.multiply(loss_mask,log_prob)  #Removing label 0(Feasible but incorrect transitions) softmax_logits from log-loss
        batch_loss= tf.reduce_sum(correct_logits, axis=1)   #Summation of Correct Transitions
        loss = - tf.reduce_mean(batch_loss)  # Final negative loss.
        regularization = self._regularization_lambda * (tf.nn.l2_loss(self.weights1) + tf.nn.l2_loss(self.bias) + tf.nn.l2_loss(self.weights2) + tf.nn.l2_loss(self.tokens))    #Computing regularisation l2 loss for each parameter

        # TODO(Students) End
        return loss + regularization
