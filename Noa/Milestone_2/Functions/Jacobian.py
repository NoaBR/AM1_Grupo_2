from numpy import array, zeros


def Jacobian(G: callable, x: array, h: float) -> array:
    n = len(x)
    J = zeros((n, n))
    for i in range(n):
        e = zeros(n)
        e[i] = 1
        x_plus = x + h * e
        x_minus = x - h * e
        J[:, i] = (G(x_plus) - G(x_minus)) / (2 * h)
    return J
