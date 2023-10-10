import tensorflow as tf
import tensorflow.keras.backend as K

POS_WEIGHT = 33

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    
    # This function is from StackExchange.
    # Original contributor tobigue: https://stats.stackexchange.com/users/9440/tobigue
    # Last edited by dudelstein: https://stats.stackexchange.com/users/102166/dudelstein
    # Original URL: https://stats.stackexchange.com/questions/261128/neural-network-for-multi-label-classification-with-large-number-of-classes-outpu
    # WayBack archived version: https://web.archive.org/web/20230927190311/https://stackoverflow.com/questions/42158866/neural-network-for-multi-label-classification-with-large-number-of-classes-outpu
    
    # transform back to logits
    _epsilon = K.epsilon()
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)