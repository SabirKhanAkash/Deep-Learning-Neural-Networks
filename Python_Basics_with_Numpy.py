test = "Hello World"

print ("test: " + test)

import math
from public_tests import *
import numpy as np

# GRADED FUNCTION: basic_sigmoid

def basic_sigmoid(x):
    s = 1/(1+ math.exp(-x))

    return s


print("basic_sigmoid(1) = " + str(basic_sigmoid(1)))

basic_sigmoid_test(basic_sigmoid)
x = [1, 2, 3] 
basic_sigmoid(x) 


t_x = np.array([1, 2, 3])
print(np.exp(t_x)) 

t_x = np.array([1, 2, 3])
print (t_x + 3)

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    print(s)
    
    return s

t_x = np.array([1, 2, 3])
print("sigmoid(t_x) = " + str(sigmoid(t_x)))

sigmoid_test(sigmoid)


# GRADED FUNCTION: sigmoid_derivative

def sigmoid_derivative(x):
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    
    return ds

t_x = np.array([1, 2, 3])
print ("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))

sigmoid_derivative_test(sigmoid_derivative)

# GRADED FUNCTION:image2vector

def image2vector(image):

    image.shape
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2],1)
    
    return v

t_image = np.array([[[ 0.67826139,  0.29380381],
                     [ 0.90714982,  0.52835647],
                     [ 0.4215251 ,  0.45017551]],

                   [[ 0.92814219,  0.96677647],
                    [ 0.85304703,  0.52351845],
                    [ 0.19981397,  0.27417313]],

                   [[ 0.60659855,  0.00533165],
                    [ 0.10820313,  0.49978937],
                    [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(t_image)))

image2vector_test(image2vector)

# GRADED FUNCTION: normalize_rows

def normalize_rows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm

    return x

x = np.array([[0, 3, 4],
              [1, 6, 4]])
print("normalizeRows(x) = " + str(normalize_rows(x)))

normalizeRows_test(normalize_rows)

# GRADED FUNCTION: softmax

def softmax(x):

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1,keepdims=True)
    s = x_exp/x_sum
    
    return s

t_x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(t_x)))

softmax_test(softmax)

# GRADED FUNCTION: L1

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    #(≈ 1 line of code)
    # loss = 
    # YOUR CODE STARTS HERE
    loss = np.sum(abs(y-yhat))
    
    # YOUR CODE ENDS HERE
    
    return loss


# In[32]:


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))

L1_test(L1)


# <a name='ex-9'></a>
# ### Exercise 9 - L2
# Implement the numpy vectorized version of the L2 loss. There are several way of implementing the L2 loss but you may find the function np.dot() useful. As a reminder, if $x = [x_1, x_2, ..., x_n]$, then `np.dot(x,x)` = $\sum_{j=0}^n x_j^{2}$. 
# 
# - L2 loss is defined as $$\begin{align*} & L_2(\hat{y},y) = \sum_{i=0}^{m-1}(y^{(i)} - \hat{y}^{(i)})^2 \end{align*}\tag{7}$$

# In[33]:


# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    #(≈ 1 line of code)
    # loss = ...
    # YOUR CODE STARTS HERE
    loss = np.dot(y-yhat,y-yhat)
    
    # YOUR CODE ENDS HERE
    
    return loss


# In[34]:


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2(yhat, y)))

L2_test(L2)


# Congratulations on completing this assignment. We hope that this little warm-up exercise helps you in the future assignments, which will be more exciting and interesting!

# <font color='blue'>
# <b>What to remember:</b>
#     
# - Vectorization is very important in deep learning. It provides computational efficiency and clarity.
# - You have reviewed the L1 and L2 loss.
# - You are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc...
