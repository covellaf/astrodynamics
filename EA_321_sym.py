# =============================================================================
# Created By  : Francesca Covella
# Created Date: Wednesday 09 June 2021
# =============================================================================

"""
Using the Python simbolic notation to code the 321 Euler angle rotation.
"""

from sympy import symbols, expand, factor, Matrix, cos, sin, simplify, det, solve

alpha1=symbols('alpha1')
alpha2=symbols('alpha2')
alpha3=symbols('alpha3')
alpha1_dot=symbols('alpha1_dot')
alpha2_dot=symbols('alpha2_dot')
alpha3_dot=symbols('alpha3_dot')
A_om_B = symbols('A_om_B')
det_om_matrix = symbols('det_om_matrix')

R1 = Matrix( [[1, 0, 0], [0, cos(alpha1), sin(alpha1)], [0, -sin(alpha1), cos(alpha1)]] )
print(f'The matrix about axis {3} of angle alpha1 is \n{R1} and the dimensions are {R1.shape}')
R2  = Matrix( [[cos(alpha2), 0, -sin(alpha2)], [0, 1, 0], [sin(alpha2), 0, cos(alpha2)]] )
print(f'The matrix about axis {2} of angle alpha2 is \n{R2} and the dimensions are {R2.shape}')
R3 = Matrix( [[cos(alpha3), sin(alpha3), 0], [-sin(alpha3), cos(alpha3), 0], [0, 0, 1]] )
print(f'The matrix about axis {1} of angle alpha3 is \n{R3} and the dimensions are {R3.shape}')

prime_basis = (R2**(-1)) * (R3**(-1))    # Matrix( [(R2**(-1)) * (R3**(-1))] )
prime_basis = simplify(prime_basis)
print('After simplifying ------------------------')
print(prime_basis)

A_om_B = alpha3_dot * (prime_basis[2, :]).T + alpha2_dot * (prime_basis[1, :]).T + alpha1_dot * Matrix([1, 0, 0])
A_om_B = simplify(A_om_B)
print('------------------------ A omega B (A is the fixed ref frame and B is the perifocal ref frame) ')
print(A_om_B)

om_matrix = Matrix([[1, 0,           -sin(alpha2)],
                    [0, cos(alpha1), cos(alpha2)*sin(alpha1)],
                    [0, -sin(alpha1), cos(alpha1)*cos(alpha2)] 
                    ])
simplify(om_matrix**(-1))
det_om_matrix = simplify(om_matrix.det())
print(f'The determinant of the matrix A omega B is {det_om_matrix}')
sing = solve(det_om_matrix)
print(f'Singulaities occurs for a value of {sing[0]} or {sing[1]} about the second rotation.')