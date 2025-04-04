import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Определение переменной и функций
t = sp.symbols('t')
a, b = -1.7, 1.7
f = 8 - t**2
y = sp.sin(2*t)

# Ортогонализация методом Грама-Шмидта
def gram_schmidt(x_funcs, n):
    phi = []
    for i in range(n):
        phi_i = x_funcs[i]
        for j in range(i):
            phi_i -= sp.integrate(x_funcs[i] * phi[j] * f, (t, a, b)) / sp.integrate(phi[j]**2 * f, (t, a, b)) * phi[j]
        phi.append(phi_i)
    return phi

# Создание системы функций x_n(t) = t^{n-1}
n_max = 5  # Максимальное количество функций для ортогонализации
x_funcs = [t**n for n in range(n_max)]
phi = gram_schmidt(x_funcs, n_max)

# Вывод ортогонализированных функций
print("Ортогонализированные функции:")
for i, p in enumerate(phi):
    p_simplified = sp.simplify(p)
    print(f"φ_{i+1}(t) = {p_simplified}")

# Вычисление коэффициентов Фурье
c = []
for p in phi:
    num = sp.integrate(y * p * f, (t, a, b))
    den = sp.integrate(p**2 * f, (t, a, b))
    c.append(num / den)

print("\nКоэффициенты Фурье:")
for i, coeff in enumerate(c):
    print(f"c_{i+1} = {sp.simplify(coeff)}")

# Построение частичных сумм ряда Фурье
def fourier_partial_sum(t_val, N):
    S_N = 0
    for n in range(N):
        S_N += c[n] * phi[n].subs(t, t_val)
    return S_N

# Построение графиков
t_vals = np.linspace(a, b, 1000)
y_vals = np.array([sp.N(y.subs(t, tv)) for tv in t_vals])

# Графики для разных N
N_list = [1, 3, 5]
plt.figure(figsize=(12, 6))
plt.plot(t_vals, y_vals, label='y(t) = sin(2t)', linewidth=2)

for N in N_list:
    S_N_vals = np.array([sp.N(fourier_partial_sum(tv, N)) for tv in t_vals])
    plt.plot(t_vals, S_N_vals, '--', label=f'N = {N}')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.title('Приближение функции y(t) частичными суммами ряда Фурье')
plt.show()
