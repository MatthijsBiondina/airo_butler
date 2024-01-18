from typing import List, Tuple
import numpy as np

from pairo_butler.utils.tools import pyout


def compute_triangle_angles_sss(a: float, b: float, c: float):
    assert a > 0, f"Cannot compute triangle with side <a> of length 0"
    assert b > 0, f"Cannot compute triangle with side <b> of length 0"
    assert c > 0, f"Cannot compute triangle with side <c> of length 0"
    if b + c == a:
        alpha = np.pi
    elif a + b == c or a + c == b:
        alpha = 0
    else:
        alpha = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

    if a + c == b:
        beta = np.pi
    elif b + a == c or b + c == a:
        beta = 0.0
    else:
        beta = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))

    if a + b == c:
        gamma = np.pi
    elif c + a == b or c + b == a:
        gamma = 0.0
    else:
        gamma = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

    return alpha, beta, gamma


def find_intersections_between_circles(
    x0: float, y0: float, r0: float, x1: float, y1: float, r1: float
) -> List[Tuple[float, float]]:
    distance_between_centers = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if distance_between_centers > (r0 + r1):
        # No solution, the circles are separate
        return []
    if distance_between_centers < abs(r0 - r1):
        # No solution, one circle is contained within the other.
        return []
    if distance_between_centers == 0 and r0 == r1:
        # No solution, the circles are coincident.
        return []

    # Calculate intersection points
    a = (r0**2 - r1**2 + distance_between_centers**2) / (
        2 * distance_between_centers
    )
    h = np.sqrt(r0**2 - a**2)
    x2 = x0 + a * (x1 - x0) / distance_between_centers
    y2 = y0 + a * (y1 - y0) / distance_between_centers

    x3a = x2 + h * (y1 - y0) / distance_between_centers
    y3a = y2 - h * (x1 - x0) / distance_between_centers
    x3b = x2 - h * (y1 - y0) / distance_between_centers
    y3b = y2 + h * (x1 - x0) / distance_between_centers

    if x3a == x3b and y3a == y3b:
        return [(x3a, y3a)]
    else:
        return [(x3a, y3a), (x3b, y3b)]
