""" This is a file for utility functions for multiple modules """
import tensorflow as tf

def make_summary_vec(name, tensor):
    """ tensor shape: (1, size). 
        Returns:
            a list of tf.summary.ops
    """
    ops = []
    if len(tensor.shape) == 2:
        for i in range(tensor.get_shape()[0]):
            for j in range(tensor.get_shape()[1]):
                ops.append(tf.summary.scalar(name + '_' + str(i) + '_' + str(j), tensor[i, j]))
    elif len(tensor.shape) == 1:
        for i in range(tensor.shape[0]):
            ops.append(tf.summary.scalar(name + '_' + str(i), tensor[i]))
    else:
        assert False, "It only handles tensors one-dim or two-dimensional"
    return ops

def make_summary_vec_np(name, array):
    """ array shape: (1, size). 
        Returns:
            a tf Summary
    """
    summary_list = []
    if len(array.shape) == 2:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                summary_list.append(tf.Summary.Value(tag=name + '_' + str(i) + '_' + str(j), simple_value=array[i,j]))
    elif len(array.shape) == 1:
        for i in range(array.shape[0]):
            summary_list.append(tf.Summary.Value(tag=name + '_' + str(i), simple_value=array[i]))
    else:
        assert False, "It only handles tensors one-dim or two-dimensional"
    return tf.Summary(value=summary_list)
