import sys
from typing import Callable
from functions import f_grad, g_grad, h_grad




def gradient_descent(
    gradient: Callable[[float, float], tuple[float, float]],
    x0: float,
    y0: float,
    learning_rate: float,
    precision: float,
    max_iter: int = 1000000,
    debug: bool = False,
) -> tuple[float, float, int]:
    '''
        Perform gradient descent to minimize f starting from (x0, y0).
    '''
    x = x0
    y = y0
    iter_count = 0
    while True:
        iter_count += 1
        if iter_count > max_iter:
            print("Maximum iterations reached, couldn't converge.")
            break
        df_dx, df_dy = gradient(x, y)
        if debug:
            print(f"Iteration {iter_count}: x = {x}, y = {y}, df/dx = {df_dx}, df/dy = {df_dy}")
        x_new = x - learning_rate * df_dx
        y_new = y - learning_rate * df_dy

        if abs(x_new - x) < precision and abs(y_new - y) < precision:
            break

        x = x_new
        y = y_new
    return x, y, iter_count

def gradient_ascent(
    gradient: Callable[[float, float], tuple[float, float]],
    x0: float,
    y0: float,
    learning_rate: float,
    precision: float,
    max_iter: int = 1000000,
    debug: bool = False,
) -> tuple[float, float, int]:
    '''
        Perform gradient ascent to maximize f starting from (x0, y0).
    '''
    x = x0
    y = y0
    iter_count = 0
    while True:
        iter_count += 1
        if iter_count > max_iter:
            print("Maximum iterations reached, couldn't converge.")
            break
        df_dx, df_dy = gradient(x, y)
        if debug:
            print(f"Iteration {iter_count}: x = {x}, y = {y}, df/dx = {df_dx}, df/dy = {df_dy}")
        x_new = x + learning_rate * df_dx
        y_new = y + learning_rate * df_dy

        if abs(x_new - x) < precision and abs(y_new - y) < precision:
            break

        x = x_new
        y = y_new
    return x, y, iter_count

if __name__ == "__main__":
    if len(sys.argv) != 6 and len(sys.argv) != 7:
        print("Usage: python main.py <function> <x0> <y0> <learning_rate> <precision> (optional: debug true/false)")
        sys.exit(1)
    f_name = sys.argv[1]
    type_opt = "ascend" if f_name == "h" else "descend"
    x0 = float(sys.argv[2])
    y0 = float(sys.argv[3])
    learning_rate = float(sys.argv[4])
    precision = float(sys.argv[5])
    debug = True if len(sys.argv) == 7 and sys.argv[6].lower() == "true" else False
    if f_name == "f":
        print("Minimizing function f")
        func_grad = f_grad
    elif f_name == "g":
        print("Minimizing function g")
        func_grad = g_grad
    elif f_name == "h":
        print("Minimizing function h")
        func_grad = h_grad
    else:
        print(f"Function {f_name} not recognized.")
        sys.exit(1)
    if type_opt == "descend":
        x, y, i = gradient_descent(func_grad, x0, y0, learning_rate, precision, debug=debug)
    else:
        x, y, i = gradient_ascent(func_grad, x0, y0, learning_rate, precision, debug=debug)
    print(f"x = {x}, y = {y}, iterations = {i}")