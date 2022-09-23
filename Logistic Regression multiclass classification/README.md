# Logistic-Regression-One-vs-All-using-gradient-descent

In this project I tried to implement Gradient descent algorithm for Logistic Regression One Vs All method on the Iris Data set. The entire data set is divided into three different classes. Our main target was to build a model which takes some charactersictics of the flower as input and predict what species it is.
The Gradient descent is based on the minimization of the cost function by finding suitable parameters expressed by vector $\theta$.
In a Binary classification Model we only need one set of parameter vectors $\theta$. Whereas in Multi class classification we need set of parameter vectors equal to number of class labels.
## Brief theory :
In logistic regression we define a hypothesis by means of sigmoid function.    $h_\theta(z)=\frac{1}{1+e^{-z}}$.
where $z=X\theta$. 
$\theta$ is column vector if its binary classification. 
In case of Multiclass classification we can take $\theta$ as a matrix with each column for particular class label. 

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/97800241/178110476-7678861a-0fb8-497c-bde0-54b6752e78bc.gif)


![CodeCogsEqn (2)](https://user-images.githubusercontent.com/97800241/178110705-e8f2a362-341d-4843-bfa2-dd8b5d18d874.gif).

0<y<1. Gradient descent function is same as the linear regression parameter update. So a label encoded target must be converted to Binary one hot encoded matrix. Its a very important step. We get indivual costs for each class. 

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/97800241/178110965-bf200a92-45cb-4dae-aa0f-b389e9c3066f.gif).
Learning rate for our Code is set to 0.01.
