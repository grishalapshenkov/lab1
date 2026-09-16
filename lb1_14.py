"""
Лабораторная работа №1. Вариант 14.
Итерационные методы вычисления sqrt(a), cbrt(b), ln(c).
Критерий остановки для корней: по невязке |F(x)| < eps.
Дополнительное требование Д4: ограничение N_max, диагностика при
недостижении точности, поиск минимального достаточного N_max.
"""
import math


class PrecisionNotReached(Exception):
    """Точность не достигнута за отведённое число итераций."""
    def __init__(self, n_max, last_x, last_residual):
        self.n_max = n_max
        self.last_x = last_x
        self.last_residual = last_residual
        super().__init__(
            f"Точность не достигнута за N_max={n_max} итераций "
            f"(последнее x={last_x!r}, невязка={last_residual:.3e})"
        )


def sqrt_newton(a, eps=1e-6, n_max=100):
    """Квадратный корень методом Ньютона (формула Герона).
    Критерий остановки — по невязке: |x_new^2 - a| < eps."""
    if a < 0:
        raise ValueError("Подкоренное выражение должно быть неотрицательным")
    if a == 0:
        return 0.0, 0
    x = a if a >= 1 else 1.0
    for n in range(1, n_max + 1):
        x_new = 0.5 * (x + a / x)
        residual = abs(x_new * x_new - a)
        if residual < eps:
            return x_new, n
        x = x_new
    raise PrecisionNotReached(n_max, x, residual)


def cbrt_newton(b, eps=1e-6, n_max=100):
    """Кубический корень методом Ньютона (с учётом знака).
    Критерий остановки — по невязке: |x_new^3 - |b|| < eps."""
    if b == 0:
        return 0.0, 0
    sign = -1 if b < 0 else 1
    bb = abs(b)
    x = bb if bb >= 1 else 1.0
    for n in range(1, n_max + 1):
        x_new = (2 * x + bb / (x * x)) / 3
        residual = abs(x_new ** 3 - bb)
        if residual < eps:
            return sign * x_new, n
        x = x_new
    raise PrecisionNotReached(n_max, sign * x, residual)


def ln_series(c, eps=1e-6, n_max=10_000):
    """Натуральный логарифм рядом ln c = 2*(y + y^3/3 + y^5/5 + ...),
    y = (c-1)/(c+1). Критерий остановки — по модулю члена ряда: |t_k| < eps."""
    if c <= 0:
        raise ValueError("Аргумент логарифма должен быть положительным")
    y = (c - 1) / (c + 1)
    y2 = y * y
    power = y
    total = 0.0
    for k in range(n_max):
        term = 2 * power / (2 * k + 1)
        if abs(term) < eps:
            return total, k
        total += term
        power *= y2
    raise PrecisionNotReached(n_max, total, abs(term))


def find_min_n_max(func, arg, eps):
    """Подбор минимального N_max, достаточного для достижения eps."""
    n = 1
    while True:
        try:
            _, iters = func(arg, eps, n)
            return iters
        except PrecisionNotReached:
            n += 1


if __name__ == "__main__":
    a, b, c = 37, -125, 12
    eps = 1e-6
    N_MAX = 5  # ограничение по Д4

    print("=== Расчёт с ограничением N_max = 5 ===\n")
    for name, func, arg in [("sqrt(37)", sqrt_newton, a),
                             ("cbrt(-125)", cbrt_newton, b),
                             ("ln(12)", ln_series, c)]:
        try:
            val, n = func(arg, eps, N_MAX)
            print(f"{name}: результат={val}, итераций={n}")
        except PrecisionNotReached as e:
            print(f"{name}: ОШИБКА — {e}")

    print("\n=== Минимальное достаточное N_max (eps = 1e-6) ===\n")
    for name, func, arg in [("sqrt(37)", sqrt_newton, a),
                             ("cbrt(-125)", cbrt_newton, b),
                             ("ln(12)", ln_series, c)]:
        print(f"{name}: N_max(min) = {find_min_n_max(func, arg, eps)}")

    print("\n=== Итоговый расчёт с достаточным N_max ===\n")
    header = f"{'Функция':<12}{'Аргумент':>10}{'Результат':>18}{'Эталон':>18}{'Погрешность':>14}{'Итераций':>10}"
    print(header)
    for name, func, arg, ref in [
        ("sqrt", sqrt_newton, a, math.sqrt(a)),
        ("cbrt", cbrt_newton, b, math.copysign(abs(b) ** (1 / 3), b)),
        ("ln", ln_series, c, math.log(c)),
    ]:
        val, n = func(arg, eps, 1000)
        err = abs(val - ref)
        print(f"{name:<12}{arg:>10}{val:>18.10f}{ref:>18.10f}{err:>14.2e}{n:>10}")