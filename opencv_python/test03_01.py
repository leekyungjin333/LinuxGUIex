import numpy as np

a = np.zeros((2, 2))
print("np.zeros")
print(a)
print("")

a = np.ones((2, 3))
print("np.ones")
print(a)
print("")

a = np.full((2, 3), 5)
print("np.full")
print(a)
print("")

a = np.eye(3)
print("np.eye")
print(a)
print("")

a = np.array(range(20)).reshape((4, 5))
print(a)
