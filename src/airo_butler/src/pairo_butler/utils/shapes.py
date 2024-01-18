import numpy as np


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
