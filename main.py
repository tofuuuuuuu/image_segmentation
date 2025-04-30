import cv2
import numpy as np
import matplotlib.pyplot as plt
import graph 
from sklearn.neighbors import NearestNeighbors

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
blur = cv2.GaussianBlur(gray, (3, 3), 0)

grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
grad = np.sqrt(grad_x**2 + grad_y**2)
grad_norm = (255 * grad / grad.max())

n, m = grad_norm.shape
lowerbound = np.percentile(grad_norm, 80) 
edge_nodes = []

# take the higher 20% brightness
for i in range(n) :
    for j in range(m) :
        if lowerbound <= grad_norm[i, j] :
            edge_nodes.append([i, j])

points = np.array(edge_nodes)
sample_idx = np.random.choice(len(points), size=min(len(points), 100000), replace=False) # sample 10^5 points
sample = points[sample_idx]
k = 10
knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(sample)

t = graph.DSU(n * m)

print("adding edges")

for p1 in sample :
    distance, idx = knn.kneighbors([p1])
    for j in range(k):
        p2 = sample[idx[0][j]]
        c1 = coord_to_int(p1[0], p1[1], n, m)
        c2 = coord_to_int(p2[0], p2[1], n, m)
        if c1 < c2 : 
            t.addEdge(graph.Edge(c1, c2, distance[0][j]))

print("done adding edges")

print("finding mst")

mst = t.kruskal()

print("done mst")

# TODO: hair removal

clean_mst = graph.Hair_Remover(n * m, mst)
clean_edgeset = clean_mst.remove_hairs(4) # remove hairs of order 4 (arbitrary value)

# TODO: path compression 

lines_mst = []
for e in mst :
    c1 = int_to_coord(e.a, n, m)
    c2 = int_to_coord(e.b, n, m)
    lines_mst.append((c1, c2))


lines_clean = []
for e in clean_edgeset :
    c1 = int_to_coord(e[0], n, m)
    c2 = int_to_coord(e[1], n, m)
    lines_clean.append((c1, c2))



print("graphing")

plt.subplot(3, 3, 1)
plt.imshow(inp)
plt.axis('off')
plt.title('Input')

plt.subplot(3, 3, 2)
plt.imshow(gray, cmap = 'gray')
plt.axis('off')
plt.title('Gray Scale')

plt.subplot(3, 3, 3)
plt.imshow(blur, cmap = 'gray')
plt.axis('off')
plt.title('Gaussian Blur')

plt.subplot(3, 3, 4)
plt.imshow(grad_norm)
plt.axis('off')
plt.title('Edges')

plt.subplot(3, 3, 5)
for l in lines_mst:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('MST')

plt.subplot(3, 3, 6)
for l in lines_clean:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('Hair removal')

plt.subplot(3, 3, 7)
for l in lines_clean:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.imshow(inp)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('Overlay')

print("done graphing")

plt.show()