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
    
class Hair_Remover :
    def __init__(self, n, edges) :
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.posterity_rank = [0] * n
        self.vis = [False] * n
        self.remove = [False] * n
        for e in edges :
            if e.a == e.b :
                continue
            self.adj[e.a].append(e.b)
            self.adj[e.b].append(e.a)

    def __dfs_posterity(self, v, prev):
        mx = 0
        for i in self.adj[v] :
            if i == prev : 
                continue
            self.__dfs(i, v)
            mx = max(mx, self.posterity_rank[i])
        self.posterity_rank[v] = mx + 1
    
    def __dfs_mark_remove(self, v, prev, h) :
        self.vis[v] = True
        
        if self.posterity_rank[v] <= h :
            if self.posterity_rank[prev] > h + 1 :
                self.remove[v] = True
                for i in self.adj[v] :
                    if i == prev : 
                        continue
                    self.remove[i] = True
            elif len(self.adj[v]) > 2 :
                for i in self.adj[v] : 
                    if i == prev :
                        continue
                    self.remove[i] = True
                first = self.adj[v][0]
                second = self.adj[v][1]
                if first == prev :
                    self.remove[second] = False
                else :
                    self.remove[first] = False

        for i in self.adj[v] :
            if i == prev :
                continue
            self.__dfs_mark_remove(i, v, h)

    def remove_hairs(self, h) :
        rt = 0 # root at 0 (arbitrary value)
        self.__dfs_posterity(rt, rt)
        for i in range(self.n) :
            if self.vis[i] :
                continue
            self.__dfs_mark_remove(i, i, h)

        edges = []
        for i in range(self.n) : 
            for j in self.adj[i] :
                if i >= j or self.remove[i] or self.remove[j]: 
                    continue
                edges.append([i, j])
        return edges

        