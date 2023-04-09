import sympy as s

x1, y1, z1, x2, y2, z2, a, b = s.symbols('x1 y1 z1 x2 y2 z2 a b', real=True, positive=True)
r1 = s.sqrt(x1**2 + y1**2 + z1**2)
r2 = s.sqrt(x2**2 + y2**2 + z2**2)
psi = s.exp(-a*r1)*s.exp(-a*r2)

# print(s.latex(psi))
# print(s.latex(s.diff(psi, x1, 1)))
# print(s.latex(s.diff(psi, x1, 2)))

r12 = s.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
jastrow = s.exp(b*r12)
# print(s.latex(jastrow))
# print(s.latex(s.diff(jastrow, x1, 1)))
# print(s.latex(s.diff(jastrow, x1, 2)))

lap = s.diff(jastrow, x1, 2) + s.diff(jastrow, y1, 2) + s.diff(jastrow, z1, 2)
print(s.latex(s.simplify(lap)))
