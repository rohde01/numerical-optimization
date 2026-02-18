from matplotlib import pyplot as plt
import numpy as np
from case_studies import *

#uncomment for environments ike WSL that have no gui
#import matplotlib
#matplotlib.use('agg')


def conjugate_gradients(Q,g,epsilon):
    """
    Implements the Conjugate Gradient function that finds the approximate minimum of the function

    f(x) = 1/2 x^TQx + g^Tx

    Arguments:
    Q: A nxn positive definite matrix
    g: A n vector
    epsilon: the tolerance, the algorithm terminates when |grad f(x)|<epsilon

    Returns:
    x_k: the approximate optimum 
    x_k_history: A list of solution candidates x_k computes by CG
    """
    x_k = np.zeros(g.shape[0]) #algorithm starts at 0
    x_k_history=[x_k.copy()] #history of x values computed
    
    # Implement Algorithm 5 here. 
    # Remember to .append() your solution candidates x_k to the list.
        
    return x_k, x_k_history

#setup problem 
n=50
epsilon=1.e-10
Q=np.random.randn(n,n//2)
Q=Q@Q.T +np.eye(n)
g = np.random.randn(n)

#analytical solution
solution = -np.linalg.inv(Q)@g 

#run algorithm
solution_cg, history = conjugate_gradients(Q,g,epsilon)

#test for correctness
print("numerical difference:", np.linalg.norm(solution-solution_cg))
grad_cg_sol = Q@solution_cg + g
print("grad norm:", np.linalg.norm(grad_cg_sol))

#plot the history
history_grad = [np.linalg.norm(Q@x + g) for x in history]
plt.semilogy(history_grad)
plt.ylabel("Gradient norm")
plt.xlabel("Iterations")
plt.savefig("results_cg.png")
