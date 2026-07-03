import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(7)

x = ai.Parameter((3, 1), requires_grad=False)
W = ai.Parameter((3, 3), init="xavier")
b = ai.Parameter((3, 1), init="zeros")

y = (W @ x) + b
loss = (y ** 2).mean()

print("x:")
print(x)

print("\nW:")
print(W)

print("\ny:")
print(y)

print("\nloss:")
print(loss)

loss.backward()

print("\nGradient of W:")
print(W.grad)

print("\nGradient of b:")
print(b.grad)
