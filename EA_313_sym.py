# =============================================================================
# Created By  : Francesca Covella
# Created Date: Wednesday 09 June 2021
# =============================================================================

"""
Using the Python simbolic notation to code the 313 Euler angle rotation.

NOTE on SymPy:
We can manipulate irrational numbers exactly using SymPy
sympy.sqrt(8) = 2*sqrt(2)
Symbolic computation systems (which by the way, are also often called computer algebra systems, 
or just CASs) such as SymPy are capable of computing symbolic expressions with variables.
Unlike many symbolic manipulation systems, variables in SymPy must be defined before they are used.
x, y = symbols('x y')
expr = x + 2*y
expr - x = 2*y
However most simplifications are not performed automatically
expr * x = x*(x + 2*y) and not x**2 + 2*x*y 
But there are functions to go from one form to the other
expand(expr*x) = x**2 + 2*x*y and factor(expr*x) = x*(x + 2*y)
SymPy can simplify expressions, compute derivatives, integrals, and limits, 
solve equations, work with matrices, it includes modules for plotting, printing,
code generation, physics, statistics, combinatorics, number theory, geometry, logic, and more.
e.g.
x = symbols('x')
diff(sin(x)*exp(x), x)
integrate(exp(x)*sin(x) + exp(x)*cos(x), x)
integrate(sin(x**2), (x, -oo, oo))
solve(x**2 - 2) = [-sqrt(2), sqrt(2)]
t = symbols('t')
y = Function('y')
dsolve(Eq(y(t).diff(t, t) - y(t), exp(t)), y(t)) = Eq(y(t), C2*exp(-t) + (C1 + t/2)*exp(t))
Matrix([[1, 2], [2, 2]]).eigenvals()
You can also find the Latex expression by using
latex(Integral(cos(x)**2, (x, 0, pi)))
"""

# import sympy
from sympy import symbols, expand, factor, Matrix, cos, sin, simplify, det, solve
# init_printing(use_unicode=True)
"""
NOTE on Matrices:
A matrix is constructed by providing a list of row vectors that make up the matrix.
A list of elements is considered to be a column vector.
Unlike every other object in SymPy, they are mutable. This means that they can be modified in place, 
as we will see below. The downside to this is that Matrix cannot be used in places that require immutability, 
such as inside other SymPy expressions or as keys to dictionaries. If you need an immutable version of Matrix, 
use ImmutableMatrix.
Some Basic operations:
shape gives the dimension of the matrix (row, col)
To get an individual row or column of a matrix, use row or col. e.g. matrix.col(#col)
To delete a row or column, use row_del or col_del. e.g. matrix.col_del(#col)
These operations will modify the Matrix in place.
To insert rows or columns, use row_insert or col_insert. e.g. M = M.row_insert(1, Matrix([[0, 4]]))
In general, a method that does not operate in place will return a new Matrix 
and a method that does operate in place will return None.
simple operations like addition and multiplication are done just by using +, *, and **. 
To find the inverse of a matrix, just raise it to the -1 power.
To take the transpose of a Matrix M, use T. e.g. M.T
To create an identity matrix, use eye. eye(n) will create an 𝑛×𝑛 identity matrix.
To create a matrix of all zeros, use zeros. zeros(n, m) creates an 𝑛×𝑚 matrix of 0s.
ones creates a matrix of 1s.
To create diagonal matrices, use diag. The arguments to diag can be either numbers or matrices.
To compute the determinant of a matrix, use the det method. e.g. M.det()
https://docs.sympy.org/latest/tutorial/matrices.html
"""

inclination = symbols('inclination') 
RAAN = symbols('RAAN') 
arg_per = symbols('arg_per')
RAAN_dot = symbols('RAAN_dot') 
inclination_dot = symbols('inclination_dot')
arg_per_dot = symbols('arg_per_dot')
A_om_B = symbols('A_om_B')
det_om_matrix = symbols('det_om_matrix')

R3_Om = Matrix( [[cos(RAAN), sin(RAAN), 0], [-sin(RAAN), cos(RAAN), 0], [0, 0, 1]] )
print(f'The matrix about axis {3} of angle Omega is \n{R3_Om} and the dimensions are {R3_Om.shape}')
R1_i  = Matrix( [[1, 0, 0], [0, cos(inclination), sin(inclination)], [0, -sin(inclination), cos(inclination)]] )
print(f'The matrix about axis {1} of angle i is \n{R1_i} and the dimensions are {R1_i.shape}')
R3_om = Matrix( [[cos(arg_per), sin(arg_per), 0], [-sin(arg_per), cos(arg_per), 0], [0, 0, 1]] )
print(f'The matrix about axis {3} of angle omega is \n{R3_om} and the dimensions are {R3_om.shape}')

prime_basis = (R1_i**(-1)) * (R3_om**(-1))    # Matrix( [(R1_i**(-1)) * (R3_om**(-1))] )
prime_basis = simplify(prime_basis)
print('After simplifying ------------------------')
print(prime_basis)

A_om_B = RAAN_dot * (prime_basis[2, :]).T + inclination_dot * (prime_basis[0, :]).T + arg_per_dot * Matrix([0, 0, 1])
A_om_B = simplify(A_om_B)
print('------------------------ A omega B (A is the fixed ref frame and B is the perifocal ref frame) ')
print(A_om_B)

om_matrix = Matrix([[sin(arg_per)*sin(inclination),  cos(arg_per), 0], 
                    [cos(arg_per)*sin(inclination), -sin(arg_per), 0], 
                    [cos(inclination),                0,           1]])
         
simplify(om_matrix**(-1))
det_om_matrix = simplify(om_matrix.det())
print(f'The determinant of the matrix A omega B is {det_om_matrix}')
sing = solve(det_om_matrix)
print(f'Singulaities occurs for a value of {sing[0]} or {sing[1]} about the second rotation.')