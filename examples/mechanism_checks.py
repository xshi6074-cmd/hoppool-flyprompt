"""Synthetic algebra examples; no training, benchmark data, or model traces.

Run with Python 3.10+. Only the standard library is needed.
These checks illustrate identities and counterexamples in the README.
They do not establish accuracy, retention, or performance of a real model.
"""

import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def attention(logits, values):
    shifted = [math.exp(x - max(logits)) for x in logits]
    weights = [x / sum(shifted) for x in shifted]
    return [sum(w * v[j] for w, v in zip(weights, values))
            for j in range(len(values[0]))]


def check_attention_partition():
    # A fixed query's scaled key scores; dropout is disabled.
    original_scores = [0.2, -0.7, 1.1]
    prompt_scores = [0.8, -0.3]
    original_values = [[1.0, 0.0], [0.3, -0.2], [-0.1, 0.8]]
    prompt_values = [[-0.3, 1.2], [0.9, 0.4]]
    mass_x = sum(math.exp(s) for s in original_scores)
    mass_p = sum(math.exp(s) for s in prompt_scores)
    gamma = mass_p / (mass_x + mass_p)
    ax = attention(original_scores, original_values)
    ap = attention(prompt_scores, prompt_values)
    direct = attention(original_scores + prompt_scores,
                       original_values + prompt_values)
    decomposed = [(1 - gamma) * x + gamma * p for x, p in zip(ax, ap)]
    error = max(abs(x - y) for x, y in zip(direct, decomposed))
    assert error < 1e-12

    # Even a freely chosen scalar cannot add a non-collinear value component.
    # One original value [1, 0], one prompt value [0, 1], equal key scores.
    mixed = attention([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    original = [1.0, 0.0]
    best_scalar = dot(mixed, original) / dot(original, original)
    residual = math.sqrt(sum((y - best_scalar * x) ** 2
                             for x, y in zip(original, mixed)))
    assert math.isclose(residual, 0.5, abs_tol=1e-12)
    print("PASS: attention partition; scalar-only gate counterexample")


def check_affine_second_moments():
    # Four deterministic points have mean mu and covariance L L^T.
    # This is an exact finite-distribution construction, not Gaussian sampling.
    mu = [0.4, -0.2]
    columns = [[1.0, 0.8], [0.0, 0.6]]
    delta_w = [[1.0, 1.0], [-0.3, 0.2]]
    delta_b = [0.1, -0.4]
    points = [[m + sign * math.sqrt(2) * c for m, c in zip(mu, col)]
              for col in columns for sign in (-1, 1)]
    explicit = sum(sum((dot(row, z) + b) ** 2
                       for row, b in zip(delta_w, delta_b))
                   for z in points) / len(points)
    mean_term = sum((dot(row, mu) + b) ** 2
                    for row, b in zip(delta_w, delta_b))
    trace_term = sum(dot(row, col) ** 2 for row in delta_w for col in columns)
    assert math.isclose(explicit, mean_term + trace_term, abs_tol=1e-12)

    # Covariance [[1, .8], [.8, 1]], versus its diagonal, for delta_w=[1,1].
    correlated = sum(dot([1.0, 1.0], col) ** 2 for col in columns)
    diagonal = 2.0
    assert math.isclose(correlated, 3.6, abs_tol=1e-12)
    assert not math.isclose(correlated, diagonal)
    print("PASS: affine second-moment identity; correlation can matter")


def check_nonlinear_boundary():
    # Both distributions have mean 0 and variance 1. For g(z)=z^2 and
    # teacher=0, squared discrepancy is z^4 and their expectations differ.
    distribution_a = [(-1.0, 0.5), (1.0, 0.5)]
    distribution_b = [(0.0, 0.5), (-math.sqrt(2), 0.25), (math.sqrt(2), 0.25)]

    def moment(distribution, power):
        return sum(probability * z ** power for z, probability in distribution)

    for power in (1, 2):
        assert math.isclose(moment(distribution_a, power),
                            moment(distribution_b, power), abs_tol=1e-12)
    assert math.isclose(moment(distribution_a, 4), 1.0, abs_tol=1e-12)
    assert math.isclose(moment(distribution_b, 4), 2.0, abs_tol=1e-12)
    print("PASS: equal first/second moments do not determine nonlinear loss")


if __name__ == "__main__":
    print("Synthetic algebra only; no experimental performance evidence.")
    check_attention_partition()
    check_affine_second_moments()
    check_nonlinear_boundary()
