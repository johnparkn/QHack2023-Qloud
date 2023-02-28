from matplotlib import pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
############################################
# Define Problem
qubit_n = 10
u = 2
n = 2
k = 2
C = 3
w = [1, 2]
pen = 1


####################################################
# Make Optimization Probelm (QUBO)

Q = np.zeros((qubit_n,qubit_n))

# 1st term of QUBO
Q1 = np.zeros((qubit_n,qubit_n))
for i in range(n):
    # yi
    Q1[i,i] = 1


# 2nd term of QUBO
Q2u = np.zeros((qubit_n,qubit_n, u))
for i in range(u):
    vect2 = np.zeros((qubit_n,1))
    vect2[i] = C
    
    for j in range(n):
        xij = n + i*u+j
        vect2[xij] = -w[j]
        
    for j in range(k):
        bij = n + u*n + u*i+j
        vect2[bij] = -2**(j)
        
    Q2u[:,:,i] = np.matmul(vect2, vect2.T)
 
    
# 3rd term of QUBO
Q3n = np.zeros((qubit_n,qubit_n, n))
for j in range(n):
    vect3 = np.zeros((qubit_n,1))
    for i in range(u):
        xij = n + i*u+j
        vect3[xij] = 1
    Q3n[:,:,j] = np.matmul(vect3, vect3.T)
    
    for i in range(u):
        xij = n + i*u+j
        Q3n[xij,xij,j] =Q3n[xij,xij,j]-2
        
        
Q = Q1
for i in range(u):
    Q = Q + pen * Q2u[:,:,i]
for j in range(n):
    Q = Q + pen* Q3n[:,:,j]
    

# Define Problem in Gurobi formation
m = gp.Model()
x = m.addVars(qubit_n, vtype=GRB.BINARY, name="x")
obj=gp.quicksum( gp.quicksum( Q[ii,jj] * x[ii] for ii in range(qubit_n) ) * x[jj] for jj in range(qubit_n) )
m.setObjective(obj, GRB.MINIMIZE)

######################################################
# Optimize algorothm

start = time.time()
m.optimize()
    
end = time.time()

print(f"{end - start:.5f} sec")


# Show Optimization result
classical_solution = np.array( m.x)
cl1 = np.matmul(classical_solution.T , Q)
cl2 = np.matmul(cl1, classical_solution)
print("classical output : ",classical_solution)

