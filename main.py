import sys
from typing import Callable
from functions import f_grad, g_grad, h_grad, obj_f, obj_g, obj_h


def backtracking_line_search_max(
    gradient: Callable[[float, float], tuple[float, float]],
    obj_f: Callable[[float, float], float],
    x: float,
    y: float,
    initial_lr: float,
    alpha: float = 0.3,
    beta: float = 0.8,
    max_iter: int = 50,
    min_lr: float = 1e-12,
) -> float:
    '''
        Perform backtracking line search to find an appropriate learning rate for decreasing the objective function.
    '''
    lr = initial_lr
    df_dx, df_dy = gradient(x, y)
    current_value = obj_f(x, y)
    it = 0
    while it < max_iter and lr > min_lr:
        new_x = x - lr * df_dx
        new_y = y - lr * df_dy
        new_value = obj_f(new_x, new_y)
        # Armijo condition (sufficient decrease)
        if new_value <= current_value - alpha * lr * (df_dx**2 + df_dy**2):
            return lr
        lr *= beta
        it += 1
    # fallback: return the (possibly reduced) lr
    return lr   
                       



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

def gradient_descent_variable(
    gradient: Callable[[float, float], tuple[float, float]],
    obj_f: Callable[[float, float], float],
    x0: float,
    y0: float,
    learning_rate: float,
    precision: float,
    max_iter: int = 1000000,
    debug: bool = False,
) -> tuple[float, float, int]:
    '''
        Perform gradient descent to minimize f starting from (x0, y0) with a variable step size.
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
        # compute step-size using backtracking line search (Armijo)
        lr = backtracking_line_search_max(gradient, obj_f, x, y, learning_rate, alpha=0.3, beta=0.8)
        x_new = x - lr * df_dx
        y_new = y - lr * df_dy

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
    if len(sys.argv) != 7 and len(sys.argv) != 8:
        print(len(sys.argv))
        print("Usage: python main.py <function> <type> <x0> <y0> <learning_rate> <precision> (optional: debug true/false)")
        sys.exit(1)
    f_name = sys.argv[1]
    type_opt = sys.argv[2]
    if type_opt == "des":
        type_opt = "descend"
        print("Using descend")
    elif type_opt == "des_var":
        type_opt = "descend_var"
        print("Using descend with variable step size")
    elif type_opt == "asc":
        type_opt = "ascend"
        print("Using ascend")
    else:
        print(f"Type {type_opt} not recognized.")
        sys.exit(1)
    x0 = float(sys.argv[3])
    y0 = float(sys.argv[4])
    learning_rate = float(sys.argv[5])
    precision = float(sys.argv[6])
    debug = True if len(sys.argv) == 8 and sys.argv[7].lower() == "true" else False
    if f_name == "f":
        print("Minimizing function f")
        func_grad = f_grad
        obj_function = obj_f
    elif f_name == "g":
        print("Minimizing function g")
        func_grad = g_grad
        obj_function = obj_g
    elif f_name == "h":
        print("Minimizing function h")
        func_grad = h_grad
        obj_function = obj_h
    else:
        print(f"Function {f_name} not recognized.")
        sys.exit(1)
    if type_opt == "descend":
        x, y, i = gradient_descent(func_grad, x0, y0, learning_rate, precision, debug=debug)
    elif type_opt == "descend_var":
        x, y, i = gradient_descent_variable(func_grad, obj_function, x0, y0, learning_rate, precision, debug=debug)
    elif type_opt == "ascend":
        x, y, i = gradient_ascent(func_grad, x0, y0, learning_rate, precision, debug=debug)
    print(f"x = {x}, y = {y}, iterations = {i}")
    with open("result.txt", "a") as f:
        f.write(f"Function: {f_name}\n")
        f.write(f"Type: {type_opt}\n")
        f.write(f"Initial point: ({x0}, {y0})\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"X={x} Y={y} Iterations={i}\n")
        f.write("-" * 50 + "\n")
        