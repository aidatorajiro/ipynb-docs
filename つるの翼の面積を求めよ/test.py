from sympy import *
from sympy.geometry import *

A = Point(0, 0) # 1
B = Point(1, 0) # 2
C = Point(0, 1) # 3
D = Point(1, 1) # 4

f = Segment(C, D) # 5
g = Segment(D, B) # 6
h = Segment(B, A) # 7
i = Segment(A, C) # 8
p = Segment(A, D) # 9
l = Segment(C, B) # 10

j = Segment(C, A).perpendicular_bisector() # 11
k = Segment(C, D).perpendicular_bisector() # 12

E = intersection(k,f)[0] # 13
F = intersection(j,k)[0] # 14
G = intersection(j,i)[0] # 15
H = intersection(k,h)[0] # 16
I = intersection(j,g)[0] # 17

m = Line(Triangle(E, C, F).bisectors()[C]) # 18

J = intersection(m,k)[0] # 19

n = Line(Triangle(E, C, J).bisectors()[C]) # 20
q = p.parallel_line(J) # 21

K = intersection(n, q)[0] # 22

r = k.parallel_line(K) # 23
s = Line(Triangle(E, D, F).bisectors()[D]) # 24

K_ = K.reflect(f) # 25
L = intersection(n,s)[0] # 26
L_ = L.reflect(f) # 27

Poly = Polygon(K_, K, L, D, L_)

print(latex(Poly.area))
