"""Contains mathematical utility functions.

Defines functions for law of sines and cosines for solving SAS
triangles.
"""
from math import asin, cos, sin, sqrt


def cosines_law(b: float, c: float, A: float) -> float:
    """Calculates missing side length using law of cosines."""
    return sqrt(b**2 + c**2 - 2(a)(b)(cos(A)))


def sines_law(a: float, b: float, B: float) -> float:
    """Calculates missing angle using law of sines."""
    return asin(a * sin(B) / b)


def send_to_arduino(message: str, device: str = 'dev/ttyACM1', baud: int = 9600):
    """Sends a message from computer to connected arduino."""
    ser = serial.Serial(device, baud, timeout=1)
    ser.write(message)
    ser.flush()
