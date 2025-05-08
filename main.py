import cv2
import numpy as np
import matplotlib.pyplot as plt
import graph 
from sklearn.neighbors import NearestNeighbors
import coord

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
sample_idx = np.random.choice(len(points), size=min(len(points), 3000000), replace=False) # sample 3*10^6 points
sample = points[sample_idx]
k = 10
knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(sample)

t = graph.DSU(n * m)

print("adding edges")

for p1 in sample :
    distance, idx = knn.kneighbors([p1])
    for j in range(k):
        p2 = sample[idx[0][j]]
        c1 = coord.coord_to_int(p1[0], p1[1], n, m)
        c2 = coord.coord_to_int(p2[0], p2[1], n, m)
        if c1 < c2 : 
            t.addEdge(graph.Edge(c1, c2, distance[0][j]))

print("done adding edges")

print("finding mst")

mst = t.kruskal()
for i in range(len(mst)) :
    mst[i] = [mst[i].a, mst[i].b]

print("done mst")

print ("removing hair")

clean_mst = graph.Hair_Remover(n * m, mst)
clean_edgeset = clean_mst.remove_hairs(3) # remove hairs of order 5 (arbitrary value)

print("done removing hair")

print("compressing")

compress_mst = graph.Compressor(n * m, clean_edgeset, n, m)
compress_edgeset = compress_mst.compress(0.0005) # merge edges with cos diff by 0.0005 (arbitrary value)
# note: for larger images, choose smaller d value 

print("done compressing")

lines_mst = []
for e in mst :
    c1 = coord.int_to_coord(e[0], n, m)
    c2 = coord.int_to_coord(e[1], n, m)
    lines_mst.append((c1, c2))

lines_clean = []
for e in clean_edgeset :
    c1 = coord.int_to_coord(e[0], n, m)
    c2 = coord.int_to_coord(e[1], n, m)
    lines_clean.append((c1, c2))

lines_compress = []
for e in compress_edgeset :
    c1 = coord.int_to_coord(e[0], n, m)
    c2 = coord.int_to_coord(e[1], n, m)
    lines_compress.append((c1, c2))

print("image dimensions:", n, "x", m)
print("hair-removed edge count :", len(lines_clean))
print("final edge count :", len(lines_compress))

print("graphing")

plt.subplot(3, 3, 1)
plt.imshow(inp)
plt.axis('off')
plt.title('Input')

plt.subplot(3, 3, 2)
plt.imshow(gray, cmap = 'gray')
plt.axis('off')
plt.title('Gray scale')

plt.subplot(3, 3, 3)
plt.imshow(blur, cmap = 'gray')
plt.axis('off')
plt.title('Gaussian blur')

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
for l in lines_compress:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('Compressed')

plt.subplot(3, 3, 8)
for l in lines_clean:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.imshow(inp)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('Overlay with hair removed')

plt.subplot(3, 3, 9)
for l in lines_compress:
    x = [l[0][0], l[1][0]]
    y = [l[0][1], l[1][1]]
    plt.plot(y, x, color = 'red', linewidth = 1)
plt.imshow(inp)
plt.xlim(0, m)  
plt.ylim(n, 0) 
plt.axis('off')
plt.title('Overlay with compressed')

print("done graphing")

plt.show()