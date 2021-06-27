from math import sqrt, sin, cos, asin


def cosines_law(b: float, c: float, A: float) -> float:
    """Calculates missing side length using law of cosines."""
    return sqrt(b**2 + c**2 - 2(a)(b)(cos(A)))


def sines_law(a: float, b: float, B: float) -> float:
    """Calculates missing angle using law of sines."""
    return asin(a * sin(B) / b)
