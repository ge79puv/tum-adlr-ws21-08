import math


def joints_preprocessing(q):
    q_normalized = q / math.pi
    return q_normalized


def joints_postprocessing(q):
    q_world = q * math.pi
    return q_world

