import cfg
from ElementData import ElementData
import os
import shutil
import math

class GrafHash:

    __slots__ = ['_nodes', '_out', '_in']

    class Vertex:

        __slots__ = ['_UUID', '_valor'] 

        def __init__(self, UUID, val) -> None:
            self._UUID = UUID
            self._valor = val
        
        def __str__(self):
            return str(self._valor)

        @property
        def valor(self):
            return self._valor
        
    def __init__(self, ln=[],lv=[],lp=[], digraf = False):
        self._nodes = { } #nodes = uuid: vertex(obj:elementdata)
        self._out = { } #uuid: uuid vecino: peso
        self._in = {} if digraf else self._out
        for n in ln:
            self.insert_vertex(n)
        if lp==[]:
            for v in lv:
                self.insert_edge( v[0], v[1])
        else:
            for v,p in zip(lv,lp):
                self.insert_edge(v[0], v[1], p)
    
    @property
    def nodes(self):
        return self._nodes

    def es_digraf(self):
        return self._out!=self._in

    def __contains__(self, key):
        if key in self._nodes.keys():
            return True
        else:
            return False
    
    def __iter__(self):
        for key in self._nodes:
            yield key

    def __repr__(self):
        return ('Nodes: '+ str(len(self._nodes))) 

    def __len__(self):
        return len(self._nodes)
    
    def __getitem__(self, key):
        try: 
            return self._nodes[key].valor
        except: 
            return None

    def __delitem__(self, key):
        del self._nodes[key]


    def get(self, key): #crida a __getitem__ i li pasa un akey depenent d l'item que busqui
        return self.__getitem__(key)

    def insert_vertex(self, UUID, element): #inserta un nou vertex al graf
        if isinstance(element, ElementData):
            v = self.Vertex(UUID, element)
            self._nodes[UUID] = v
            self._out[UUID] = { }
            if self.es_digraf():
                self._in[UUID] = {}
        
    def insert_edge(self, uuid1, uuid2, p=1): #inserta un nou edge entre nodes dels parametres amb el pes corresponent
        try:
            if self._out[uuid1][uuid2] > 0:
                self._out[uuid1][uuid2] += p
        except:
            self._out[uuid1][uuid2] = p
        try:
            if self._in[uuid2][uuid1] > 0:
                self._in[uuid2][uuid1] += p
        except:
            self._in[uuid2][uuid1] = p

    def edges_out(self, UUID): #retorna un diccionari amb els nodes als que apunta
        try:
            return self._out[UUID]
        except:
            return None

    def edges_in(self, uuid): #retorna un dic amb els nodes que l'apunten
        try:
            return self._in[uuid]
        except:
            return None
    
    def minDistance(self, dist, visitat): #retorna el node a menor distanca de self, auxiliar d dijkstra
        minim= math.inf
        res=""
        for node,distancia in dist.items():
            if node not in visitat and distancia < minim:
                minim=distancia
                res=node
        return res

    def dijkstra(self,n1,n3): #retorna la distancia minima entre dos nodes utilitzant l'algoritme de dijkstra
        #inicialitza les variables
        dist={nAux:math.inf for nAux in self._out} 
        visitat = {}
        dist[n1] = 0
        predecessor = {}
        predecessor[n1]=None
        count = 0
        while count < len(self._nodes)-1 : # itera tots els nodes
            nveiAct = self.minDistance(dist, visitat)
            visitat[nveiAct] = True
            if nveiAct == n3: # si arriba al node desitjat retorna directament els diccionaris
                return dist, predecessor
            elif nveiAct in self._out:
                for n2,p2 in self._out[nveiAct].items():
                    if (n2 not in visitat):
                        if (dist[nveiAct] + p2 < dist[n2]):
                            dist[n2] = dist[nveiAct] + p2
                            predecessor[n2] = nveiAct
            count +=1
        return dist , predecessor

    def camiMesCurt(self, UUID1, UUID2): 
        cami = []
        if UUID1 in self._nodes and UUID2 in self._nodes:
            dist,predecessor=self.dijkstra(UUID1,UUID2)
            if UUID2 in predecessor:
                nodeAct = UUID2
                cami = []
                while nodeAct != UUID1:
                    cami.append(nodeAct)
                    nodeAct = predecessor[nodeAct]
                cami.append(UUID1)
                cami.reverse()
                return cami
            return []
        return []