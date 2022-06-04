import numpy as np

class Graph:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.graph = np.zeros((height, width, 3), np.uint8)
    def update_frame(self, value):
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-1,:] = self.graph[:,1:,:]
        new_graph[self.height - value:,-1,:] = 255
        self.graph = new_graph
    def get_graph(self):
        return self.graph
