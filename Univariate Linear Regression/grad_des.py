#gradient descent Algo:
import numpy as np
import matplotlib.pyplot as plt
def cost(X,y,theta):
    m=len(y)
    return (1/(2*m))*sum((np.dot(X,theta)-y)**2)

def gd(X,y,theta,alpha,iterations):
    m=len(y)
    C=np.zeros((iterations,1))
    X=np.c_[np.ones((m,1)),X]
    for i in range(iterations):
        theta[0,0]=theta[0,0]-alpha*sum((1/m)*(np.dot(X,theta)-y))
        theta[1,0]=theta[1,0]-alpha*sum((1/m)*(np.multiply((np.dot(X,theta)-y),(X[:,1].reshape(-1,1)))))
        #print(theta[0,0],theta[1,0])
        J=cost(X,y,theta)
        C[i]=J
    plt.plot(range(iterations),C)
    plt.title("Cost vs Iterations")
    plt.xlabel("Iteratons")
    plt.ylabel("Cost")
    plt.show()
    return theta
        #if i==iterations-1:
            #print("Max iterations reached")
