import coord

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

        if a == b : 
            return False

        if self.sz[a] < self.sz[b] : 
            a, b = b, a
        self.p[b] = self.p[a]
        self.sz[a] += self.sz[b]
        return True
        
    def kruskal(self):
        self.edges.sort()
        components = 0
        for e in self.edges:
            if self.merge(e.a, e.b) :
               self.mst.append(e)
               components += 1
        return self.mst 
    
class Hair_Remover :
    def __init__(self, n, edges) :
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.vis = [False] * n
        self.remove = [False] * n
        for e in edges :
            self.adj[e[0]].append(e[1])
            self.adj[e[1]].append(e[0])

    def __dfs_posterity(self, v, prev, h):
        self.vis[v] = True
        mx = 0
        for i in self.adj[v] :
            if i == prev : 
                continue
            res = self.__dfs_posterity(i, v, h)
            mx = max(mx, res)
        posterity_rank = mx + 1
        if posterity_rank <= h :
            self.remove[v] = True
        return posterity_rank

    def remove_hairs(self, h) :
        for i in range(self.n) :
            if self.vis[i] :
                continue
            self.__dfs_posterity(i, i, h)

        edges = []
        for i in range(self.n) : 
            for j in self.adj[i] :
                if i >= j or self.remove[i] or self.remove[j]: 
                    continue
                edges.append([i, j])
        return edges
    
class Compressor :
    def __init__(self, n, edges, N, M) :
        self.n = n
        self.N = N
        self.M = M
        self.adj = [[] for _ in range(n)]
        self.vis = [False] * n
        for e in edges:
            self.adj[e[0]].append(e[1])
            self.adj[e[1]].append(e[0])
        self.compr = []
    
    def __dfs(self, v, prev, head, nxt, d) :
        self.vis[v] = True
        c1 = coord.int_to_coord(head, self.N, self.M)
        c2 = coord.int_to_coord(nxt, self.N, self.M)
        c3 = coord.int_to_coord(v, self.N, self.M)
        angle1 = coord.angle(c1, c2)
        angle2 = coord.angle(c1, c3)

        newHead = head
        newNxt = nxt
        if abs(angle1 - angle2) > d :
            self.compr.append([head, prev])
            newHead = prev
            newNxt = v
        elif len(self.adj[v]) == 1 and head != v:
            self.compr.append([head, v])

        for i in self.adj[v] :
            if i == prev :
                continue
            self.__dfs(i, v, newHead, newNxt, d)

    def compress(self, d) :
        for i in range(self.n) :
            if self.vis[i] :
                continue
            self.vis[i] = True
            for j in self.adj[i] :
                self.__dfs(j, i, i, j, d)
        return self.compr 
        

        