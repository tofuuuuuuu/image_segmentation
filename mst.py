class Edge:
    def __init__(self, a, b, weight):
        self.a = a
        self.b = b
        self.w = weight

    def __lt__(self, other):
        return self.w < other.w

class DSU:
    def __init__(self, n):
        self.n = n
        self.sz = [1] * n
        self.p = [1] * n
        self.p = list(range(n))
        self.edges = []
        self.mst = []

    def addEdge(self, e):
        self.edges.append(e)

    def find(self, v):
        if self.p[v] != v : 
            self.p[v] = self.find(self.p[v])
        return self.p[v]
    
    def merge(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if(a == b) : return False

        if self.sz[a] < self.sz[b] : 
            a, b = b, a
        self.p[b] = self.p[a]
        self.sz[a] += self.sz[b]
        return True
        
    def kruskal(self):
        self.edges.sort()
        cnt = 0
        for e in self.edges:
            if cnt == self.n-1 :
                break
            if self.merge(e.a, e.b) :
               self.mst.append(e)
               cnt += 1
        return self.mst 