import keras.backend as K
import tensorflow as tf
from DegRad import rad2deg



def metric_degrees_difference(y_true, y_pred):
    """
    A Keras metric that determines the angle in degrees between two vectors in cartesian
    space.
    Only works with tensorflow
    :param y_true: tensor of shape (?,3) with actual directions
    :param y_pred: tensor of shape (?,3) with supposed directions
    :return: tensor of shape (?,1) with angles in degrees between y_true and y_pred
    """
    # first normalise the vectors so that the inproduct is equal to the cosine of the
    # angle between them
    y_true = K.l2_normalize(y_true, axis=1)
    y_pred = K.l2_normalize(y_pred, axis=1)

    # take the inproduct between the two
    inproduct = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1)

    # now the angle between them is the arccos of this inproduct
    angle = tf.acos(inproduct)

    # transform radians to degrees
    angle = rad2deg(angle)
    return K.mean(angle)