import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mst 
from sklearn.neighbors import NearestNeighbors
import random

def coord_to_int(i, j, n, m) :
    return i * m + j

def int_to_coord(a, n, m) :
    return (a // m, a % m)

inp = cv2.imread("input.jpeg")

#scale down to around 50000 pixels
n, m = inp.shape[:2]
scaling_factor = (50000 / (n * m)) **0.5
inp = cv2.resize(inp, (int(m * scaling_factor), int(n * scaling_factor)))

# gray scale + blur
inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
grad = np.sqrt(grad_x**2 + grad_y**2)
grad_norm = (255 * grad / grad.max())

n, m = grad_norm.shape
edge_nodes = []
for i in range (n):
    for j in range(m):
        if grad_norm[i, j] != 0 :
            edge_nodes.append((grad_norm[i, j], i, j))

edge_nodes.sort()
l = len(edge_nodes)
edge_nodes = edge_nodes[math.floor(0.8 * l) : l] # take brightest 30%
sample = random.sample(edge_nodes, k=min(2000, len(edge_nodes))) # sample 1000

# TODO: nearest neighbours for faster MST


t = mst.DSU(n * m)

for p1 in sample :
    for p2 in sample :
        if p1 >= p2 :
            continue
        i1 = p1[1]
        j1 = p1[2]
        i2 = p2[1]
        j2 = p2[2]
        c1 = coord_to_int(i1, j1, n, m)
        c2 = coord_to_int(i2, j2, n, m)
        dst = (i1 - i2)**2 + (j1 - j2)**2
        t.addEdge(mst.Edge(c1, c2, dst))

print("done1")
# TODO: hair removal
my_mst = t.kruskal()
print("done2")

lines = []
for e in my_mst :
    c1 = int_to_coord(e.a, n, m)
    c2 = int_to_coord(e.b, n, m)
    lines.append((c1, c2))

plt.subplot(2, 3, 1)
plt.imshow(inp)
plt.axis('off')
plt.title('Input')

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap = 'gray')
plt.axis('off')
plt.title('Gray Scale')

plt.subplot(2, 3, 3)
plt.imshow(blur, cmap = 'gray')
plt.axis('off')
plt.title('Gaussian Blur')

plt.subplot(2, 3, 4)
plt.imshow(grad_norm, cmap = 'gray')
plt.axis('off')
plt.title('Edges')

plt.subplot(2, 3, 5)
for l in lines:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('MST')

plt.subplot(2, 3, 6)
for l in lines:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.imshow(inp)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('Overlay')

plt.show()
