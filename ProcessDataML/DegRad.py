import numpy as np


def rad2deg(rad):
    """
    Convert radians to degrees
    :param rad: angle in radians
    :return: angle in degrees
    """
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180


def azimuth_zenith_to_cartestian(zenith, azimuth):
    """
    convert a direction described by a azimuth and zenith angle into a vector in
    cartesian coordinates pointing in that direction
    :param zenith: zenith in radians
    :param azimuth: azimuth in radians
    :return: x,y,z of vector in that direction
    """
    x = np.sin(zenith)*np.cos(azimuth)
    y = np.sin(zenith)*np.sin(azimuth)
    z = np.cos(zenith)
    return x,y,z


def cartestian_to_azimuth_zenith(x,y,z):
    """
    convert a vector pointing in a direction into an azimuth and zenith angle
    :param x: x component
    :param y: y component
    :param z: z component
    :return: zenith, azimuth in radians
    """
    zenith = np.arccos(z/np.sqrt(z**2+x**2+y**2))
    azimuth = np.arctan2(y,x)
    return zenith,azimuth


def angle_between_two_vectors(y1, y2):
    """
    Calculate the angle in degrees between two vectors
    :param y1: 3D vector with (?,3) dimensions
    :param y2: 3D vector with (?,3) dimensions
    :return: angle in degrees
    """
    inproduct = np.sum(np.multiply(y1, y2), axis=1)
    norm12 = np.multiply(np.linalg.norm(y1, axis=1,keepdims=True),np.linalg.norm(y2, axis=1, keepdims=True))
    angle = np.arccos(np.divide(inproduct, norm12))
    return rad2deg(angle)



