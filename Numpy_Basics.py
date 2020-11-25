# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:01:25 2020

@author: nnath
"""
import numpy as np

#Creating 1-D Array : only 1 set of square brackets
nparr_1 = np.array([1,2,3])
print(type(nparr_1))
print('1D : \n',nparr_1)

#Creating 2-D Array : 2 sets of square brackets 
nparr_2 = np.array([[1,2,3],[4,5,6],[7,8,9]]) 
print('\n2D : \n',nparr_2)

#numpy inbuilt functions 

#arange([start,]stop[,step], dtype = None) --> Returns evenly spaced values within the interval
range_func_val = np.arange(0,10)
print(range_func_val)

print(np.arange(20,31,2)) #printing even numbers in given range 20 to 30


#zeros(shape, dtype=float,order='C') --> Returns a new array of given shape and type, filled with zeros
print(np.zeros(3)) # 1D array
print(np.zeros((3,4)))# 2D Array -- zeros((rows, columns))

#ones(shape, dtype=float, order='C') --> Returns a new array of given shape and type, filled with ones
print(np.ones(3))#1D Array
print(np.ones((3,4)))#2D Array


#linespace(*args, **kwargs) --> Returns evenly spaced number between specified interval
print(np.linspace(0,5,10))

#eye(N,M=None,k=0,dtype=float,order='C') --> Creates IDENTITY MATRIX ; N=number of rows, M=Number of Columns
print(np.eye(5))
print(np.eye(5,4))

#array of random numbers (uniform distribution between 0 to 1)
print(np.random.rand(5))#1D Array
print(np.random.rand(4,5))#2D Array; 
print('*'*10)
print(np.random.random((5)))#1D Array
print(np.random.random((4,5)))#2D Array; 

'''
The only difference is in how the arguments are handled. 
With numpy.random.rand(), the length of each dimension of the output array is a separate argument. 
With numpy.random.random(), the shape argument is a single tuple
'''

#standard normal distribution or gaussion distribution
print(np.random.randn(5))#1D Array
print(np.random.randn(5,3))#2D Array
 

#randint(low, high=None,size=None,dtype='|') --> Random interger from given range 
print(np.random.randint(0,100))# ('low', 'high'] i.e. low-inclusive ; high-exclusive
print(np.random.randint(10))
print(np.random.randint(0,10,5))# 5 random integers between 0 to 10

#reshape(shape,order='C') --> Returns an array containing same data with new dimension
arr = np.arange(10)
print(arr)#1D Array
print(arr.reshape(5,2))#2D Array ; According to number of elemets reshape the array else you get ValueError

#min() max() argmin() argmax()
arr_2 = np.random.randint(0,50,10)
print(arr_2)
print(arr_2.min())
print(arr_2.max())
print(arr_2.argmin())#position of min value 
print(arr_2.argmax())#position of min value 

#shape
arr = np.array([1,2,3,4,5,6,7,8,9,10])
print(arr.shape)#(4,) --> 1D Array

arr_i = arr.reshape(2,5)
print(arr_i.shape)#(2,5) --> 2D Array

#dtype()
arr_i.dtype

#full()
print(np.full((2,3),7))#fill array of 2 by 3 with 7 value



#masking
#A = np.random.random((5,5))#creats random matrices from 0 to 1
#for matrices from -5 and 5
A = 10 * np.random.random((5,5)) - 5
print(A) 
mask = A>0
print(mask)
print(A[mask])