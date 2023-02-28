import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
import time
############################################
# Parameter
step = 100
qubit_n = 10

init_par = [0.5]

u = 2
n = 2
k = 2

C = 3
w = [1, 2]
pen = 1

dt = 0.05

####################################################
# Setting

Q = np.zeros((qubit_n,qubit_n))
Q1st = np.zeros(qubit_n)
forc = qubit_n
# 1st
b1 = np.zeros((qubit_n))
for i in range(n):
    # yi
    b1[i] = -1/2


# 2nd
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
 
    
# 3rd
Q3n = np.zeros((qubit_n,qubit_n, n))
b3n = np.zeros((qubit_n,u))
for j in range(n):
    vect3 = np.zeros((qubit_n+1,1))
    for i in range(u):
        xij = n + i*n+j
        vect3[xij] = -1/2
        vect3[forc] = u/2 -1
        
    mat3 = np.matmul(vect3, vect3.T)
    Q3n[:,:,j] = mat3[:qubit_n,:qubit_n]
    b3n[:,j] = mat3[qubit_n, :qubit_n] + mat3[:qubit_n, qubit_n] 

      

# sum        
for i in range(u):
    Q = Q + pen * Q2u[:,:,i]
for j in range(n):
    Q = Q + pen* Q3n[:,:,j]
    
Q1st = b1
for i in range(u):
    Q1st = Q1st + pen * b2u[:,i]
for j in range(n):
    Q1st = Q1st + pen * b3n[:,j]

    
# x -> 2x-1
Q = Q
Q1st = Q1st

########################################################
# Circuit
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

coeffs = coeffs / np.linalg.norm(coeffs)
cost_h = qml.Hamiltonian(coeffs, obs)


# mixer
obs = []
coeffs= []
for i in range(qubit_n):
    obs.append(qml.PauliX(i))
    coeffs.append(1)
coeffs = coeffs / np.linalg.norm(coeffs)
mixer_h = qml.Hamiltonian(coeffs, obs)



# iHdHc
obs = []
coeffs= []
for i in range(qubit_n):
    for j in range(qubit_n):
        if Q[i,j]!=0:
            if i!=j:
                obs.append( qml.PauliY(i) @ qml.PauliZ(j) )
                coeffs.append(2*Q[i,j])
                obs.append(qml.PauliZ(i) @ qml.PauliY(j))
                coeffs.append(2*Q[i,j])

for i in range(qubit_n):
    if Q1st[i]!=0:
        obs.append(qml.PauliY(i))
        coeffs.append(2*Q1st[i])
 
coeffs = coeffs / np.linalg.norm(coeffs)
iHdHc_h = qml.Hamiltonian(coeffs, obs)






# circuit
wires = range(qubit_n)
dev = qml.device('default.qubit', wires=wires)



def circuit(params, depth, **kwargs):
    for w in wires:
       qml.Hadamard(wires=w)

        
    for deptn in range(depth):
        qml.ApproxTimeEvolution(cost_h, dt , 1)
        qml.ApproxTimeEvolution(mixer_h, dt * params[deptn], 1)            



def update(params, depth):
    circuit(params, depth)
        
    return qml.expval(iHdHc_h)    
qnode = qml.QNode(update, dev)


def cost_function(params, depth):
    circuit(params, depth)

    return qml.expval(cost_h)        
qnode2 = qml.QNode(cost_function, dev)


      

###########################################################
start = time.time()
# optimization
params = [0.0]
depth_len = 1
for stepn in range(step):
    
    beta = qnode(params, depth_len)
    params.append( -beta)
    #print(beta)
    depth_len = depth_len + 1    
    
    if (stepn%10 == 0):
        cost = qnode2(params, depth_len)
        print(stepn, "th round", cost)
    
    
    
end = time.time()

print(f"{end - start:.5f} sec")


###########################################################

# Verification
print("AWS start")
s3 = ("amazon-braket-55af0bf5e8f3", "pl-63a5400a")
sv1 = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_destination_folder=s3, wires=wires, parallel=True, shots = 100000)

@qml.qnode(sv1)
def probability_circuit(params):
    circuit(params, depth_len)
    return qml.probs(wires=wires)


probs = probability_circuit(params)


plt.style.use("seaborn")
plt.bar(range(2 ** len(wires)), probs)
plt.show()

qaoa_output = bin(np.argmax(probs))
print ("FALCON output:", qaoa_output)
###########################################################
###########################################################


np.save("FALQON_result",probs)
