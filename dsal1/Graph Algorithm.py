#!/usr/bin/env python
# coding: utf-8

# In[90]:


num_nodes = 5
edges = [(0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4)]
num_nodes, len(edges)


#  Create a class to represent a graph as an adjacency list in Python

# In[91]:


for n1, n2 in edges:
    print('n1:',n1, 'n2:', n2)


# In[132]:


class Graph:
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.data = [[] for _ in range(num_nodes)]
        for n1, n2 in edges:
            self.data[n1].append(n2)
            self.data[n2].append(n1)
    
    def __repr__(self):
        [enumerate(self.data)]
            


# In[133]:


graph1 = Graph(num_nodes,edges)


# In[134]:


for x in enumerate (graph1.data):
    print(x)


# In[135]:


graph1.data


# In[136]:


for x in enumerate ([5, 3, 4, 1]):
    print(x)
    


# Represent a graph as an adjacency matrix in Python

# In[97]:


class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    def add_edge(self, v1, v2, weight=1):
        if 0 <= v1 < self.num_vertices and 0 <= v2 < self.num_vertices:
            self.adj_matrix[v1][v2] = weight
            self.adj_matrix[v2][v1] = weight  # For an undirected graph

    def display(self):
        for row in self.adj_matrix:
            print(row)


# Example usage:
num_vertices = 5
graph = Graph(num_vertices)

graph.add_edge(0, 1)
graph.add_edge(0, 3)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(2, 4)

graph.display()


# ## Graph Traversal
# 
# 
# ### Breadth-First Search
# 
# 
# 
# > **Question**: Implement breadth-first search given a source node in a graph using Python.
# 
# 
# <img src="https://i.imgur.com/E2Up1Pk.png" width="400">
# 
# BFS pseudocode (Wikipedia):
# 
# ```
#  1  procedure BFS(G, root) is
#  2      let Q be a queue
#  3      label root as discovered
#  4      Q.enqueue(root)
#  5      while Q is not empty do
#  6          v := Q.dequeue()       
#  7          for all edges from v to w in G.adjacentEdges(v) do
#  8              if w is not labeled as discovered then
#  9                  label w as discovered
# 10                  Q.enqueue(w)
# ```
# 
# 

# In[137]:


def bfs(graph, root):
    queue = []
    discovered = [False] * len(graph.data)
    distance = [None]  * len(graph.data)
    parent = [None] *  len(graph.data)
    
    discovered[root] = True
    queue.append(root)
    distance[root]=0
    idx = 0
    
    while idx < len(queue):
        #dequeue
        
        current = queue[idx]
        idx +=1
        
        #check all edges of current 
        for node in graph.data[current]:
            if not discovered[node]:
                distance [node] = 1+ distance[current]
                parent[node]= current
                discovered[node] = True
                queue.append(node)
            
    return queue, distance, parent   


# In[138]:


bfs(graph1, 3) #the output what  we get in form of visited,  node by node.


# <img src="https://i.imgur.com/E2Up1Pk.png" width="400">
# 

# In[139]:


def bfs(graph, source):
    queue = []
    discovered = [False] * len(graph.data)
    
    discovered [root] = True 
    queue.append(root)
    idx = 0
    
    while idx < len(queue):
        
        #dequeue 
        
        current = queue[idx]
        idx += 1
         
        # chek all edges of current 
        for node in graph.data[current]: # self.data[current] contains list of all nodes that is connected to current node.
            if not discovered[node]:
                discovered[node] = True
                queue.append(node)
                
        return queue


# In[140]:


bfs(graph1, 3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[102]:


from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # For an undirected graph

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            vertex = queue.popleft()
            print(vertex, end=" ")

            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)


# In[103]:


# Example usage:
graph = Graph()
graph.add_edge(0, 1)
graph.add_edge(0, 3)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(2, 4)

start_vertex = 0
print("BFS starting from vertex", start_vertex)
graph.bfs(start_vertex)


# > **Question**: Write a program to check if all the nodes in a graph are connected
# 
# 

# In[141]:


num_nodes3 = 9
edges3 = [(0, 1), (0, 3), (1, 2), (2, 3), (4, 5), (4, 6), (5, 6), (7, 8)]
num_nodes3, len(edges3)


# In[ ]:





# In[ ]:





# ## Depth-first search
# 
# 
# 
# 
# > **Question**: Implement depth first search from a given node in a graph using Python.
# 
# <img src="https://i.imgur.com/E2Up1Pk.png" width="400">
# 
# DFS pseudocode (Wikipedia):
# 
# ```
# 1.procedure DFS_iterative(G, v) is
# 2.  let S be a stack
# 3.    S.push(v)
# 4.   while S is not empty do
# 5.        v = S.pop()
# 6.       if v is not labeled as discovered then
# 7.            label v as discovered
# 8.            for all edges from v to w in G.adjacentEdges(v) do 
# 9.                S.push(w)
# ```
# 
# 
# 

# In[142]:


l1 = [5,6,2]
l1.append(3)
v = l1.pop()
v, l1


# In[143]:


def dfs(graph, root):
    stack = []
    discovered = [False] * len(graph.data)
    
    stack.append(root)
    
    while len(stack) > 0:
        current = stack.pop()
        discovered [current] = True
        result.append(current)
        for node in graph.data[current]:
            stack.append(node)

    return result


# In[144]:


graph1.data


# **Question**: Write a function to detect a cycle in a graph

# In[109]:


def has_cycle(graph):
    def dfs(node, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    num_nodes = len(graph)
    visited = [False] * num_nodes

    for node in range(num_nodes):
        if not visited[node]:
            if dfs(node, -1):
                return True

    return False

# Example usage
graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2, 4],
    4: [3]
}

if has_cycle(graph):
    print("The graph contains a cycle.")
else:
    print("The graph does not contain a cycle.")


# ### Weighted Graphs
# 
# ![](https://i.imgur.com/wy7ZHRW.png)
# 

# In[145]:


num_nodes3 = 5
edges3 = [(0,1),(1,2),(2,3),(2,4),(4,2),(3,0)]
directed = True

graph3 = Graph(num_nodes, edges, directed= True)
graph3


# ### Directed Graphs
# 
# <img src="https://i.imgur.com/8AN7EUV.png" width="480">

# In[146]:


num_nodes6 = 5
edges6 = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 2), (3, 0)]
directed6 = True
num_nodes6, len(edges6)


# > **Question**: Define a class to represent weighted and directed graphs in Python.

# In[148]:


class Graph:
    def __init__(self, num_nodes, edges, doirected= False, weighted = False):
        self.num_nodes = num_nodes
        self.weighted = weihghted
        self.directed = directed
        self.data = [[] for _ in range (num_nodes)]
        self.weight = [[] for _ in range (num_nodes)]
        for edge in edges:
            node1, node2, weight = edge
            self.data[node1].append(node2)
            self.weight[node1].append(weight)
            if not directed:
                self.data[node2].append(node1)
                self.data[node2].append(weight)
            else:
                node1, node2 = edge
                self.data[node].append(node2)
                if not directed :
                    self.data[node2].append(node1)
                    
    def __repr__(self):
        result = ""
        if self.weighted:
            for i, (nodes, weights) in enumerate(zip(self.data, self.weights)):
                result +=  "{}: {}\n".format(i,list[zip(nodes,weights)])
        else:
            for i , nodes in enumerate(self.data):
                result += "{}:{}\n".format(i, nodes)
        return result
        
          


# In[149]:


graph1= Graph(num_nodes, edges)
graph1.data


# ## Shortest Paths
# 
# 
# > **Question**: Write a function to find the length of the shortest path between two nodes in a weighted directed graph.
# 
# <img src="https://i.imgur.com/Zn5cUkO.png" width="480">
# 
# 
# **Dijkstra's algorithm (Wikipedia)**:
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)
# 
# 1. Mark all nodes unvisited. Create a set of all the unvisited nodes called the unvisited set.
# 2. Assign to every node a tentative distance value: set it to zero for our initial node and to infinity for all other nodes. Set the initial node as current.[16]
# 3. For the current node, consider all of its unvisited neighbours and calculate their tentative distances through the current node. Compare the newly calculated tentative distance to the current assigned value and assign the smaller one. For example, if the current node A is marked with a distance of 6, and the edge connecting it with a neighbour B has length 2, then the distance to B through A will be 6 + 2 = 8. If B was previously marked with a distance greater than 8 then change it to 8. Otherwise, the current value will be kept.
# 4. When we are done considering all of the unvisited neighbours of the current node, mark the current node as visited and remove it from the unvisited set. A visited node will never be checked again.
# 5. If the destination node has been marked visited (when planning a route between two specific nodes) or if the smallest tentative distance among the nodes in the unvisited set is infinity (when planning a complete traversal; occurs when there is no connection between the initial node and remaining unvisited nodes), then stop. The algorithm has finished.
# 6. Otherwise, select the unvisited node that is marked with the smallest tentative distance, set it as the new "current node", and go back to step 3.

# In[150]:


import heapq

def shortest_path_length(graph, start_node, end_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    
    priority_queue = [(0, start_node)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_node == end_node:
            return current_distance
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return float('inf')  # No path found

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}

start_node = 'A'
end_node = 'D'

length = shortest_path_length(graph, start_node, end_node)

if length < float('inf'):
    print(f"The shortest path length from {start_node} to {end_node} is {length}.")
else:
    print(f"No path found from {start_node} to {end_node}.")


# In[151]:


def update_distances(graph, current, distance, parent=None):
    """Update the distances of the current node's neighbors"""
    neighbors = graph.data[current]
    weights = graph.weight[current]
    for i, node in enumerate(neighbors):
        weight = weights[i]
        if distance[current] + weight < distance[node]:
            distance[node] = distance[current] + weight
            if parent:
                parent[node] = current

def pick_next_node(distance, visited):
    """Pick the next univisited node at the smallest distance"""
    min_distance = float('inf')
    min_node = None
    for node in range(len(distance)):
        if not visited[node] and distance[node] < min_distance:
            min_node = node
            min_distance = distance[node]
    return min_node


# In[152]:


num_nodes7 = 6
edges7 = [(0, 1, 4), (0, 2, 2), (1, 2, 5), (1, 3, 10), (2, 4, 3), (4, 3, 4), (3, 5, 11)]
num_nodes7, len(edges7)


# ### Binary Heap
# 
# A data structure to maintain the running minimum/maximum of a set of numbers, supporting efficient addition/removal.
# 
# 
# <img src="https://i.imgur.com/ABAcM7m.png" width="400">
# 
# 
# Heap operations:
# 
# - Insertion - $O(log N)$
# - Min/Max - $O(1)$ (depending on type of heap)
# - Deletion - $O(log N)$
# - Convert a list to a heap - $O(n)$
# 
# 
# Python's built-in heap: https://docs.python.org/3/library/heapq.html
# 
# > **Question**: Implement Dijkstra's shortest path algorithm using the `heap` module from Python. What is the complexity of the algorithm?

# In[155]:


import heapq

def dijkstra_shortest_path(graph, start_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    
    priority_queue = [(0, start_node)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}

start_node = 'A'

shortest_distances = dijkstra_shortest_path(graph, start_node)
print("Shortest distances:", shortest_distances)


# ## Solutions
# 
# ![](https://i.imgur.com/E2Up1Pk.png)

# In[156]:


num_nodes1 = 5
edges1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4), (1, 3)]
num_nodes1, len(edges1)


# In[157]:


num_nodes3 = 9
edges3 = [(0, 1), (0, 3), (1, 2), (2, 3), (4, 5), (4, 6), (5, 6), (7, 8)]
num_nodes3, len(edges3)


# In[158]:


num_nodes5 = 9
edges5 = [(0, 1, 3), (0, 3, 2), (0, 8, 4), (1, 7, 4), (2, 7, 2), (2, 3, 6), 
          (2, 5, 1), (3, 4, 1), (4, 8, 8), (5, 6, 8)]

num_nodes5, len(edges5)


# In[159]:


# Directed graph
num_nodes6 = 5
edges6 = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 2), (3, 0)]
num_nodes6, len(edges6)


# In[160]:


num_nodes7 = 6
edges7 = [(0, 1, 4), (0, 2, 2), (1, 2, 5), (1, 3, 10), (2, 4, 3), (4, 3, 4), (3, 5, 11)]
num_nodes7, len(edges7)


# ### Adjacency List

# In[162]:


class Graph:
    def __init__(self, num_nodes, edges):
        self.data = [[] for _ in range(num_nodes)]
        for v1, v2 in edges:
            self.data[v1].append(v2)
            self.data[v2].append(v1)
            
    def __repr__(self):
        return "\n".join(["{} : {}".format(i, neighbors) for (i, neighbors) in enumerate(self.data)])

    def __str__(self):
        return repr(self)


# In[163]:


g1 = Graph(num_nodes1, edges1)


# In[164]:


g1


# ### Breadth First Search
# 
# Complexity $O(m + n)$

# In[165]:


def bfs(graph, source):
    visited = [False] * len(graph.data)
    queue = []
    
    visited[source] = True    
    queue.append(source)
    i = 0
    
    while i < len(queue):
        for v in graph.data[queue[i]]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
        i += 1
        
    return queue


# In[166]:


bfs(g1, 3)


# In[ ]:





# ### Depth First Search

# In[167]:


def dfs(graph, source):
    visited = [False] * len(graph.data)
    stack = [source]
    result = []
    
    while len(stack) > 0:
        current = stack.pop()
        if not visited[current]:
            result.append(current)
            visited[current] = True
            for v in graph.data[current]:
                stack.append(v)
                
    return result


# In[168]:


dfs(g1, 0)


# ### Directed and Weighted Graph

# In[170]:


class Graph:
 def __init__(self, num_nodes, edges, directed=False):
     self.data = [[] for _ in range(num_nodes)]
     self.weight = [[] for _ in range(num_nodes)]
     
     self.directed = directed
     self.weighted = len(edges) > 0 and len(edges[0]) == 3
         
     for e in edges:
         self.data[e[0]].append(e[1])
         if self.weighted:
             self.weight[e[0]].append(e[2])
         
         if not directed:
             self.data[e[1]].append(e[0])
             if self.weighted:
                 self.data[e[1]].append(e[2])
             
 def __repr__(self):
     result = ""
     for i in range(len(self.data)):
         pairs = list(zip(self.data[i], self.weight[i]))
         result += "{}: {}\n".format(i, pairs)
     return result

 def __str__(self):
     return repr(self)


# In[171]:


g7 = Graph(num_nodes7, edges7, directed=True)


# In[172]:


g7


# ### Shortest Path - Dijkstra's Algorithm

# In[173]:


def update_distances(graph, current, distance, parent=None):
    """Update the distances of the current node's neighbors"""
    neighbors = graph.data[current]
    weights = graph.weight[current]
    for i, node in enumerate(neighbors):
        weight = weights[i]
        if distance[current] + weight < distance[node]:
            distance[node] = distance[current] + weight
            if parent:
                parent[node] = current

def pick_next_node(distance, visited):
    """Pick the next univisited node at the smallest distance"""
    min_distance = float('inf')
    min_node = None
    for node in range(len(distance)):
        if not visited[node] and distance[node] < min_distance:
            min_node = node
            min_distance = distance[node]
    return min_node
        
def shortest_path(graph, source, dest):
    """Find the length of the shortest path between source and destination"""
    visited = [False] * len(graph.data)
    distance = [float('inf')] * len(graph.data)
    parent = [None] * len(graph.data)
    queue = []
    idx = 0
    
    queue.append(source)
    distance[source] = 0
    visited[source] = True
    
    while idx < len(queue) and not visited[dest]:
        current = queue[idx]
        update_distances(graph, current, distance, parent)
        
        next_node = pick_next_node(distance, visited)
        if next_node is not None:
            visited[next_node] = True
            queue.append(next_node)
        idx += 1
        
    return distance[dest], distance, parent


# <img src="https://i.imgur.com/Zn5cUkO.png" width="400">

# In[174]:


shortest_path(g7, 0, 5)


# In[ ]:




