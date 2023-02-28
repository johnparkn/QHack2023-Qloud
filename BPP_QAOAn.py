import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import copy
import time
############################################
steps = 500
depth = 10

# Define Problem
qubit_n = 10

init_par = np.zeros((2,depth))
for i in range(2):
    for j in range(depth):
        init_par[i,j]=0.5

u = 2
n = 2
k = 2
C = 3
w = [1, 2]
pen = 1
norm = 1


####################################################
# Problem Setting
Q = np.zeros((qubit_n,qubit_n))
Q1st = np.zeros(qubit_n)
forc = qubit_n
# 1st term of objective
b1 = np.zeros((qubit_n))
for i in range(u):
    # yi
    b1[i] = -1/2

    
# 2nd term of objective
Q2u = np.zeros((qubit_n,qubit_n, u))
b2u = np.zeros((qubit_n,u))
for i in range(u):
    vect2 = np.zeros((qubit_n+1,1))
    
    
    vect2[i] = -C/2
    vect2[forc] = C/2
    
    for j in range(n):
        xij = n + i*n+j
        vect2[xij] = +w[j]/2
        vect2[forc] = -w[j]/2
        
    for j in range(k):
        bij = n + u*n + k*i+j
        vect2[bij] = +2**(j) /2
        vect2[forc] = -2**(j) /2
        
    mat2 = np.matmul(vect2, vect2.T)
    Q2u[:,:,i] = mat2[:qubit_n,:qubit_n]
    b2u[:,i] = mat2[qubit_n, :qubit_n] + mat2[:qubit_n, qubit_n] 
 

# sum        
for i in range(u):
    Q = Q + pen * Q2u[:,:,i]
    
Q1st = b1
for i in range(u):
    Q1st = Q1st + pen * b2u[:,i]

    

Q = Q/norm
Q1st = Q1st/norm


########################################################
# Make Quantum Circuit
wires = range(qubit_n)
dev = qml.device('default.qubit', wires=wires)    


# cost function
obs = []
coeffs= []
for i in range(qubit_n):
    for j in range(qubit_n):
        if Q[i,j]!=0:
            if i!=j:
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(Q[i,j])

for i in range(qubit_n):
    if Q1st[i]!=0:
        obs.append(qml.PauliZ(i))
        coeffs.append(Q1st[i])


cost_h = qml.Hamiltonian(coeffs, obs)



# mixer with constraint
obs = []
coeffs= []
for i in range(qubit_n):

    if i ==2 or i == 4 or i == 3 or  i == 5:
        if i ==2 or i == 3:
            obs.append(qml.PauliX(i) @ qml.PauliX(i+2))
            coeffs.append(1)
            obs.append(qml.PauliY(i) @ qml.PauliY(i+2))
            coeffs.append(1)
        
    else:
        obs.append(qml.PauliX(i))
        coeffs.append(1)        
        
mixer_h = qml.Hamiltonian(coeffs, obs)



# circuit
def circuit(params, **kwargs):
    # 11 1001 0110
    qml.RY(np.pi,0)
    qml.RY(np.pi,1)
    qml.RY(np.pi,2)
    #qml.RY(np.pi,3)
    #qml.RY(np.pi,4)
    qml.RY(np.pi,5)
    #qml.RY(np.pi,6)
    qml.RY(np.pi,7)
    qml.RY(np.pi,8)
    #qml.RY(np.pi,9)
        
    for deptn in range(depth):
        qaoa.cost_layer(params[0,deptn], cost_h)
        qaoa.mixer_layer(params[1,deptn], mixer_h)
            
        


@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h)    
    



###########################################################
# Optimization
start = time.time()
optimizer = qml.AdamOptimizer()
params = np.array(init_par, requires_grad=True)


for i in range(steps):
    params, cost = optimizer.step_and_cost(cost_function, params)
    if (i%10 == 0):
        print(i, "th round", cost)

print("Optimal Parameters:", params)
end = time.time()

print(f"{end - start:.5f} sec")


###########################################################
# Verification
print("AWS start")
s3 = ("", "")
sv1 = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_destination_folder=s3, wires=wires, parallel=True, shots = 100000)

@qml.qnode(sv1)
def probability_circuit(params):
    circuit(params)
    return qml.probs(wires=wires)


probs = probability_circuit(params)


plt.style.use("seaborn")
plt.bar(range(2 ** len(wires)), probs)
plt.show()

qaoa_output = bin(np.argmax(probs))
print ("QAOAns output:", qaoa_output)

np.save("QAOAns_result",probs) 

