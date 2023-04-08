import sympy as s

x1, y1, z1, x2, y2, z2, a = s.symbols('x1 y1 z1 x2 y2 z2 a', real=True, positive=True)
r1 = s.sqrt(x1**2 + y1**2 + z1**2)
r2 = s.sqrt(x2**2 + y2**2 + z2**2)
psi = s.exp(-a*r1)*s.exp(-a*r2)

# print(s.latex(psi))
# print(s.latex(s.diff(psi, x1, 1)))
# print(s.latex(s.diff(psi, x1, 2)))
