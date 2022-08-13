"""
Create a smaller network with lesser nodes for analysis
"""
import os

import pandas as pd


class CreateSmallNetwork:

    def __init__(self, original=None, num_nodes=5):
        self.original = original
        self.num_nodes = num_nodes

        self.raw_edges = pd.read_csv("Locations/" + original + "/edges.csv")
        self.raw_vertices = pd.read_csv("Locations/" + original + "/vertices.csv")
        self.raw_od = pd.read_csv("Locations/" + original + "/od.csv")

        self.edges = None
        self.vertices = None
        self.od = None

    def save(self, fname=None):
        self.vertices = self.raw_vertices[self.raw_vertices.vert_id < self.num_nodes]
        self.vertices = self.vertices.copy()

        self.edges = self.raw_edges[(self.raw_edges.edge_tail < self.num_nodes) &
                                    (self.raw_edges.edge_head < self.num_nodes)]
        self.edges = self.edges.copy()

        self.od = self.raw_od[(self.raw_od.origin < self.num_nodes) &
                              (self.raw_od.destination < self.num_nodes)]
        self.od = self.od.copy()

        try:
            os.mkdir("Locations/" + fname)
        except:
            pass

        self.vertices.to_csv("Locations/" + fname + "/vertices.csv")
        self.edges.to_csv("Locations/" + fname + "/edges.csv")
        self.od.to_csv("Locations/" + fname + "/od.csv")

        return None
