# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:39:43 2020

@author: nnath
"""

'''
INTRODUCTION :

Creating
    random.arrays, np.empty, np.eye, np.identity, np.zeros, np.zeros_like, np.ones, np.ones_like, np.tiu, np.tril
     
Combining
    np.hstack, np.vstack, np.concatenate, np.append

Splitting
    np.hsplit, np.vsplit, np.dsplit, np.split
    
Slicing
    Array indexing , np.compress, array masks, np.where
    
Manipulating
    np.mean, np.median, matrix multiplication, np.ravel, np.roll, np.title, np.sum
    
Export/Import
    np.save, np.load
    
Others
    np.who, np.allclose, np.ployfit
    
'''

import numpy as np

#Array Creation
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)

print(np.zeros((4,5)))#Array with zeros 
print(np.ones((4,5)))#Array with ones

print(np.empty((3,3)))#Return a new array of given shape and type, without initializing entries.; empty, unlike zeros, does not set the array values to zero

print(np.diag([1,2,3]))#Array with specified diagonals with rest values filled with 0
print(np.diag([1.2,2.5,3]))
print(np.diag([1.2,3]))

a = np.arange(12).reshape((4,3))
print(np.diag(a))
#Creating  random array
print(np.random.random((4,2)))#creats array betwwen 0 and 1 range
AA = 10 * np.random.random((2,2)) - 5#to create array of value other than 0 to 1, here, -5 to 5
print(AA)
CC = 200 * np.random.random((3,3)) -100#values from -100 to 100
print(CC)

print(np.eye(3,2))#looks like identity matrice

#Combining
B=np.zeros((3,3))
print(B)
print(np.vstack((A,B)))#number of columns has to be same else error, vertical stack

print(np.hstack((A,B)))#number of rows has to be same else error; stack A and B side by side horizontally , horizontal stack

print(np.append(A,B))
print(np.append(A,B,axis = 0))#= vstack; AXIS = 0 ---> COLUMN
print(np.append(A,B,axis = 1))#= hstack; AXIS = 1 ---> ROW

#Splitting - in equal portions ; splits needs to be a divisor of the axis shape
A.shape
np.split(A,3)#default split axis  = 0 ; splits across ROW --> np.vsplit
np.split(A,3,axis=1) #splits across column -->np.hsplit


#horizontal split - creates horizontal pieces with vertical splits --> no. of columns need to be divisible by number of splits
print(A)
print(np.hsplit(A,3))

#vertical split - Number of splits is a divisor of the number of rows; cuts on ROWS
print(np.vsplit(A,3))

#dsplit - always splits on axis =2 (3d dimension); the parameter must be a divisor of the 3rd dimension
DS=np.random.random((6,6,6))*100 -50#3D array 
DS.shape
for i in np.dsplit(DS,3):
    print(i.shape)


#Indexing  A[ROW,COLUMN]
#Accessing a particular row A[ROW_INDEX]
print(A[0])#Accessing 1st row
print(A[2])#Accessing 3rd row
#Accessing rows
A[1:]#rows from 1 row index till end
A[:1]#rows upto 1 but not incuding 1

#Accessing a particular column A[:,COLUMN_INDEX]
print(A[:,0])#Accessing 1st column
print(A[:,2])#Accessing 3rd column

#Accessing single element
print(A[0,0])
print(A[1,2])

#Shape of Array
print(A.shape)#returns a tuple (column,row)
print(np.shape(A))#same result

#Slicing A[START:STOP,START:STOP] for both ROW and Column ; *STOP ~ Up to but NOT including  
print(A[1:2,0:2])#A[Row_slicing, Column_Slicing]
print(A[1:2,:2])#Begining with 0 hence can omit 0
print(A[1:3,1:3])
print(A[1:,1:])#Till end hence can omit 3
print(A[1:,1:2])#Access 5 , 8 of array

#Masking : Get specific elements from Array
#Get elements greater than 5
print(A[A>5])#Returns a 1D array with elements greater than 5

#Linear Algebra
print(A)
print(np.transpose(A))#row becomes column and vice versa
print(A.T)#Alternative way for transpose 

#Matrix multiplication
print(np.dot(A,A))#NOTE : no of columns in 1st matrix should be equal to no of rows in 2nd matrix
print(A@A)#Alternative way for matrix multiplication

#Scalar Matrix Multiplication 
print(A)
print(5*A)
print(A/2)
print(A+3)

print(np.mean(A))#Mean of the whole matrix
print(np.mean(A,axis=0))#Mean for down the column
print(np.mean(A,axis=1))#Mean for across the row

print(np.std(A))#Standard Deviation of the whole matrix
print(np.std(A,axis=0))#Standard Deviation for down the column
print(np.std(A,axis=1))#Standard Deviation for across the row

#Linear Algebra
'''
Eigenvalues and Eigenvectors 
Let A be an n*n matrix. Scalar k is called eigenvalue of A 
if there is a non-zero vector x such that Ax = kx.
Such that vector x is called a eigenvector of A corresponding to k

NOTE :
    if k is an eigenvalue of A and x is an eigenvector belonging to k, any non-zero multiple of x will be an eigen vector
'''
print(np.linalg.eig(A))#Gives (eigenvalues, eigenvector)
'''
QR DeComposition 
'''
print(np.linalg.qr(A))
'''
Singular Value Decomposition
'''
print(np.linalg.svd(A))

#Exercise: using empty function and populating Array
initial = np.empty((1,10))
for  i in range(10):
    initial[:,i]=i
print(initial)

#np.zeros vs np.zeros_like
Z0=np.random.random((2,4))
print(Z0)
print(np.zeros_like(Z0))#Exact same size as Z0 matrix: takes argumet as argument as a matrix for the size of new matrix
print(np.zeros((3,3)))#takes size given as arguments in tuple formate

#np.ones vs np.ones_like
Z1=np.random.random((3,4))
print(Z1)
print(np.ones_like(Z1))#Exact same size as Z1 matrix: takes argumet as argument as a matrix for the size of new matrix
print(np.ones((3,3)))#takes size given as arguments in tuple formate

#np.identity vs np.eye ;np.identity(n,dtype=None) ; np.eye(N,M=None,k=0,dtype=float,order='C')
print(np.identity(3))#3X3 matrix; square matrix always
print(np.eye(3))#same result as above 
print(np.eye(3,2))#row, column
print(np.eye(3,3,1))#moves the index of the diagonal up with 1
print(np.eye(3,3,-1))#moves the index of the diagonal down with 1

#np.tril --> (lower triangle), np.triu  --> (upper triangle)
'''
mid main diagonal = 0 
lower triangle i.e below the diagonal is -ve
upper triangle i.e above the diagonal is +ve
'''
TL = np.array([[-6,6,3,3],[7,3,-4,1],[1,10,1,3],[6,-1,-2,1]]) 
print(np.tril(TL))#upper triangle to 0
print(np.tril(TL,1))#1 diagonal above from main diagonal and  make upper triangle 0
print(np.tril(TL,-1))#we are keeping only 3 diagonals and rest to 0 i.e 0 in main diagonal and above triangle
print(np.tril(TL,-2))#2 diagonals away from main diagonal is to be kept ; rest to 0

print(np.triu(TL))#lower triangle to 0
print(np.triu(TL,1))#1 diagonal above from main diagonal and  make lower triangle 0
print(np.triu(TL,-1))#we are keeping only 3 diagonals and rest to 0 i.e 0 in main diagonal and below triangle
print(np.triu(TL,-2))#2 diagonals away from main diagonal is to be kept ; rest to 0


#np.concatenation
a=np.random.random((2,2))*10 - 5 
a=a.astype(int)
print(a)
b=np.random.random((2,1))*10 - 5 
b=b.astype(int)
print(b)
c=np.random.random((1,2))*10 - 5 
c=c.astype(int)
print(c)
'''
NOTE : 
Matrix shape:    
    a - 2*2
    b - 2*1
    c - 1*2
    
np.concatenation()
    Default axis = 0 ; hence need to have same number of columns 
'''
np.concatenate((a,c))#same number of columns ; as axis =0 by default
np.concatenate((a,b), axis=1)#rows have to be same dimension for axis =1 and this horizontally stacks the matrix 

#np.append()
np.append(a,b)#puts both array elements into a 1D array
np.append(a,c,axis  = 0 )#column has to be of same dimension , vertical stack
np.append(a,b,axis  = 1 )#rows has to be of same dimension , horizontal stack

#Array Masking 
print(A)
mask = A>5
print(mask)
A[mask]#Returns 1D array of all values of A that satisfies the condition 

#compress()
