"""
@author-Aayushi Srivastava
Compares 3 triangulation algorithm MDV, LB-Triang and LEx-M just on the basis of edges added and run time.
"""

import random
import numpy as np

import itertools
import copy

import networkx as nx
import matplotlib.pyplot as plt
import sys
import timeit

#from time import sleep
#import abdc
#from Queue import Queue
#from threading import Thread


class ChordalVert:
	def __init__(self, noNodes, noEdges, m_vert=0):
		"""function to initialize the variables in the instance of a ChordalGraph"""
		self.noNodes = noNodes
		self.noEdges = noEdges
		self.vertexList = []
		self.GEdgeList = []
		self.HEdgeList = [] #HEdgeList
		self.REdgeList = []
		self.G = {}
		self.H = {}
		self.R = {}
		self.neb = [] 
		self.K = {}
		self.KEdgeList = []
		self.m_vert = m_vert
		self.minv = {}
		self.neb = []
		self.NEdgeList = []
		self.LEdgeList = []
		self.fc = 0
		#self.sc = 0
		#self.alpha = [] #list for number of vertices
		self.unnumberedVertices = [] #list for vertices yet to be numbered
		self.numberedVertices = [] #list for the vertices numbered
		self.LabelsDict = {} #Dictionary with vertex as key and labels as value
		self.NumberDict = {} #Dictionary with vertexas key and number as value


		self.meo = []#Minimal elimination ordering
		self.Fill = []#Fill edges

	def ArbitraryGraph(self):
		"""function to create arbitrary graph"""
		self.G = nx.dense_gnm_random_graph(self.noNodes, self.noEdges)
		#self.G = {0: [15], 1: [8, 2, 3, 10], 2: [1, 10, 5, 6], 3: [1, 15], 4: [5], 5: [2, 4, 6, 15], 6: [9, 2, 12, 5], 7: [8, 11, 13], 8: [1, 14, 7], 9: [10, 6, 15], 10: [1, 2, 12, 13, 9], 11: [7], 12: [10, 13, 6], 13: [10, 15, 12, 7], 14: [8, 15], 15: [0, 3, 5, 9, 13, 14]}

		if type(self.G) is not dict:
			self.G = nx.to_dict_of_lists(self.G)
				
		for i in range(0, self.noNodes):
			self.vertexList.append(i)
		for key, value in self.G.iteritems():
			for v in value:
				if key<v:
					e = []
					e.append(key)
					e.append(v)
					self.GEdgeList.append(e)
		
		self.G = nx.Graph(self.G)
		connComp = sorted(nx.connected_components(self.G))
		self.G = nx.to_dict_of_lists(self.G)
		
		connComp = list(connComp)
		noOFConnComp = len(connComp)
		if noOFConnComp > 1:
			#print "Here we are"
			#print connComp
			self.G = nx.Graph(self.G)
			self.plotArbitraryGraph(self.G)
			j = 0
			while j < noOFConnComp - 1:
				u = random.choice(list(connComp[j%noOFConnComp]))
				v = random.choice(list(connComp[(j+1)%noOFConnComp]))
				self.addAnEdge(self.G, self.GEdgeList, u, v)
				j = j + 1
		#print str(self.G)
		self.G = nx.Graph(self.G)
		self.plotArbitraryGraph(self.G)
		#print "see"
		self.G = nx.to_dict_of_lists(self.G)

		 
	def addAnEdge(self, graphToAdd, edgeListToAdd, v1, v2):
		"""function to add an edge in the graph"""
		graphToAdd = nx.to_dict_of_lists(graphToAdd)
		graphToAdd[v1].append(v2)
		graphToAdd[v2].append(v1)
		e = []
		e.append(v1)
		e.append(v2)
		edgeListToAdd.append(e)
		

	def plotArbitraryGraph(self, graphToDraw):
		graphToDraw = nx.to_dict_of_lists(graphToDraw)
		#print "HEY"
		#print type(graphToDraw)
		edges = 0
		for node, degree in graphToDraw.iteritems():
			edges += len(degree) 
		print type(self.G)
		print self.G
	
		
		GD = nx.Graph(graphToDraw)
		pos = nx.spring_layout(GD)
		#print "\nArbitrary Graph: "+str(self.G)
		#print "\nNo. of edges in the Arbitrary Graph: "+ str(edges/2)
		#plt.title("Arbitrary Graph")
		#nx.draw(GD, pos, width=8.0,alpha=0.5,with_labels = True)
		nx.draw_networkx_edges(GD, pos, width=1.0, alpha=0.5)
		nx.draw_networkx_nodes(GD, pos, node_color='red', node_size=300, alpha=0.8)
		nx.draw_networkx_labels(GD,pos)
		plt.draw()
		plt.show()
		self.createChrdG()
	
	def createChrdG(self):
		"""To start MDV"""
		#sleep(5)
		starttime = timeit.default_timer()
		self.HEdgeList = copy.deepcopy(self.GEdgeList)
		self.H = copy.deepcopy(self.G)
		self.H = nx.Graph(self.H)

		print "Start Minimum Vertex Process"
		self.H = nx.Graph(self.H)
		self.Minvertex(self.vertexList,self.HEdgeList,self.H)
		self.FinalGraph(self.NEdgeList,self.vertexList)
		print "End Minimum Vertex Process"
		print "Number of Edges added in MDV: ",len(self.NEdgeList)
		print "Runtime of MDV method:",timeit.default_timer() - starttime
		#self.workLT()
		return True


	def Minvertex(self,vertexList,edgeList, graphtoCons):
		"""MDV"""
		graphtoCons = nx.Graph(graphtoCons)
		self.H = nx.Graph(self.H)
		#isChordal = False
		#print "My vertex list",vertexList
		random.shuffle(vertexList)
		self.H = nx.Graph(self.H)
		for v in vertexList:
			#print type(self.H)
			self.H = nx.Graph(self.H)
			dv = list(self.H.degree(self.H)) #list of tuples
			#dv = list(graphtoCons.degree(graphtoCons)) 
			#print "see the  degree list:"
			#print dv
			#print self.HEdgeList
			dvdict = dict(dv)
			#print "Dictionary of node-degree is", dvdict
			self.minv = dict(sorted(dvdict.items(), key=lambda kv:(kv[1], kv[0])))
			#print "Sorted dictionary of node-degree:",self.minv
			#graphtoCons = nx.to_dict_of_lists(graphtoCons)
			self.H = nx.Graph(self.H)
			#print "The dictionary looks like:", self.H
			mincp = copy.deepcopy(self.minv)
			try:
				for key,value in mincp.iteritems():
					if value < 2:
						self.minv.pop(key)
				#print "Deleted"
				#print "Updates:",self.minv
				graphtoCons = nx.Graph(graphtoCons)
				#nodeH = graphtoCons.nodes()
				nodeH = self.H.nodes()
				#print "Old Nodes are:",nodeH
				#print "New nodes are",list(self.minv)
				self.H.add_nodes_from(list(self.minv))
				self.H.remove_nodes_from(list(list(set(nodeH) - set(list(self.minv)))))
				self.H = nx.to_dict_of_lists(self.H)
				#print "New Dictionary:",self.H
				self.m_vert = min(self.minv.keys(), key=(lambda k:self.minv[k]))
				#print type(self.m_vert)
				#print "Minimum degree vertex is:",self.m_vert
				self.H = nx.Graph(self.H)
				#print "The chosen Minimum vertex is", self.m_vert
				
				self.neb = list(self.H.neighbors(self.m_vert))
				#print "Neighbors of the chosen vertex are:",self.neb
				neblen = len(self.neb)
				
				self.H = nx.Graph(self.H)
				self.H.remove_node(self.m_vert)
				self.neighbcomp(self.m_vert,self.H)
				self.H = nx.Graph(self.H)
			except ValueError as e:
				#print "Dictionary is Empty now"
				break


	def neighbcomp(self,chosvert,graphtoRecreate):
		"""to add eges amongst neighbors"""
		self.H = nx.Graph(self.H)
		nebcomb = list(itertools.combinations(self.neb,2))
		#print "See combinations:",nebcomb
		for p in nebcomb:
			v1 =  p[0]
			v2 = p[1]
			#print p
			if self.H.has_edge(*p) :
				#print p
				#print "Already edge is there"
				continue
			else:
				self.H.add_edge(*p)
				#print "Check this"
				self.NEdgeList.append(p)
				#print "My list", self.NEdgeList
				continue
		#print "Edges added using Minimum Degree",len(self.NEdgeList)

		self.H= nx.to_dict_of_lists(self.H)
		#print "See change",self.H

	def FinalGraph(self,newaddedgelist,vertexlist):
		#isChordal = False
		print "EdgeList verifying",newaddedgelist
		print "Total Edges added in Minimum Degree Process is ",len(newaddedgelist)
		GD = nx.Graph(self.G)
		pos = nx.spring_layout(GD)

		B = copy.deepcopy(self.G)
		B = nx.Graph(B)
		B.add_nodes_from(vertexlist)
		B.add_edges_from(newaddedgelist)
		B = nx.to_dict_of_lists(B)
		print "see B", B
		##Recognition----
		graph = nx.Graph(B)
		if nx.is_chordal(graph):
			print "IT IS CHORDAL"
		else :
			print "NO IT IS NOT CHORDAL"
		nx.draw_networkx_nodes(GD, pos, nodelist=vertexlist, node_color='red', node_size=300, alpha=0.8,label='Min degree')
			
		nx.draw_networkx_edges(GD, pos, width=1.0, alpha=0.5)
		nx.draw_networkx_edges(GD, pos, edgelist=newaddedgelist, width=8.0, alpha=0.5, edge_color='blue',label='Min degree')
		nx.draw_networkx_labels(GD,pos)
		plt.draw()
		plt.show()
		#plt.show(block=False)
		self.workLT()

	def createAuxGraph(self, graph, auxNodes):
		"""function to create induced graph on the set of vertices"""
		auxGraph = {}
		for i in auxNodes:
			if i in graph:
				auxGraph[i] = list(set(graph[i]).intersection(set(auxNodes)))
		return auxGraph


	def workLT(self):
		"""to start LB-Triang"""
		starttime = timeit.default_timer()
		self.REdgeList = copy.deepcopy(self.GEdgeList)
		self.R = copy.deepcopy(self.G)
		print "Start of LB_Triang"
		self.LB_Triang(self.vertexList, self.REdgeList, self.R)
		self.LBFinalGraph(self.LEdgeList, self.vertexList)
		print "End of LB-Triang"
		print "Number of Edges added in LB-Triang", len(self.LEdgeList)
		print "Runtime of LB-Triang method:",timeit.default_timer() - starttime
		#self.LChGr()
		#return True
	
	def LB_Triang(self, vertexList, edgeList, graphToRecognize):
		"""This function is implemented based on the algorithm LB-Triang from the paper "A WIDE-RANGE EFFICIENT ALGORITHM FOR 
		MINIMAL TRIANGULATION" by Anne Berry for recognition chordal graphs and add edges (if necessary) by making each vertex 
		LB-simplicial.""" 
		print "LBR"
		#graphToRecognize = nx.Graph(graphToRecognize)
		self.R = nx.Graph(self.R)
		random.shuffle(vertexList)
		#vertexVisibility = [0]*len(vertexList)
		#isChordal = False
		for v in vertexList:
			print "Goto"
			print type(self.R)
			#print "The vertex "+str(vertexList.index(v))+"-"+str(v)+" is verifying..."
			#openNeighbors = graphToRecognize[v]
			#self.R = nx.to_dict_of_lists(self.R)
			openNeighbors = list(self.R[v])

			print "My openNeighbor is:" ,openNeighbors
			#self.R = nx.to_dict_of_lists(self.R)
			closedNeighbors = copy.deepcopy(openNeighbors)
			print "CNN",closedNeighbors
			print type(closedNeighbors)
			closedNeighbors.append(v)
			print "My closed neighbors",closedNeighbors
			cNMinusE = list(set(vertexList).difference(set(closedNeighbors))) #V-S
			#print "cNMinusE is",cNMinusE
			if cNMinusE:
				#print "Loopys"
				#VMinusSGraph = self.createAuxGraph(graphToRecognize, cNMinusE) #G(V-S)
				VMinusSGraph = self.createAuxGraph(self.R, cNMinusE) #G(V-S)
				componentsOri = sorted(nx.connected_components(nx.Graph(VMinusSGraph)))
				#print "Component(s) in the graph: "+str(componentsOri)
				componentsCompAll = []
				for co in componentsOri:
					openNCO = []
					for v1 in co:
						#print type(self.R)
						#openNV1 = graphToRecognize[v1]
						openNV1 = list(self.R[v1])
						#print type(openNV1)
						#print "openNV1:",openNV1
						openNCO = openNCO+openNV1
						#print "pehle wala openNCO",openNCO
					openNCO = list(set(openNCO).difference(co))
					#print "see openNCO",openNCO
					self.LbEdges(openNCO)
					self.R = nx.to_dict_of_lists(self.R)
			#else:
				#print "The vertex "+str(v)+" does not generate any minimal separator."
				#print "================================================"
			
	def LbEdges(self,vlist):
		"""to add edges in LBT"""
		self.R = nx.Graph(self.R)
		lbcomb = list(itertools.combinations(vlist,2))
		#print "See combinations:",lbcomb
		for p in lbcomb:
			#print p
			v1 = p[0]
			v2 = p[1]
			if self.R.has_edge(*p):
				#print p
				#print "Already edge is there"
				continue
			else:
				self.R.add_edge(*p)
				#print "Check this"
				self.LEdgeList.append(p)
				#print "My list", self.LEdgeList
				templist = []
				templist.append(v1)
				templist.append(v2)
				self.REdgeList.append(templist)
					
		

	def LChGr(self):
		"""Start LEXM"""
		starttime = timeit.default_timer()
		self.K = copy.deepcopy(self.G)
		self.KEdgeList = copy.deepcopy(self.GEdgeList)
		self.unnumberedVertices = copy.deepcopy(self.vertexList)
		print "Start LexM"
		self.lexm(self.K ,self.KEdgeList)#LexM Function
		self.FinalLGraph(self.G,self.Fill,self.vertexList)
		self.finalDisplay()
		print "End Lex M"
		print "Number of Edges Added in LEx-M", len(self.Fill)
		print "Runtime of Lex-M method:",timeit.default_timer() - starttime

		return True

	def LBFinalGraph(self,newaddedgelist,vertexlist):
		GD = nx.Graph(self.G)
		pos = nx.spring_layout(GD)
		print "New edges added are:",newaddedgelist
		print "Total Edges added in LB-Triang is", len(newaddedgelist)
		F = copy.deepcopy(self.G)
		F = nx.Graph(F)
		F.add_nodes_from(vertexlist)
		F.add_edges_from(newaddedgelist)
		F = nx.to_dict_of_lists(F)
		#print "see F", F
		graph = nx.Graph(F)
		if nx.is_chordal(graph):
			print "IT IS CHORDAL"
		else:
			print "NO IT IS NOT CHORDAL"
	

		nx.draw_networkx_nodes(GD, pos,nodelist=vertexlist, node_color='red', node_size=300, alpha=0.8,label='LB_Triang')
					
		nx.draw_networkx_edges(GD, pos, width=1.0, alpha=0.5)
		nx.draw_networkx_edges(GD, pos, edgelist=newaddedgelist, width=8.0, alpha=0.5, edge_color='blue',label='LB_Triang')
		nx.draw_networkx_labels(GD,pos)
		#nx.draw_networkx(GD,pos,True)
		plt.show()
		self.LChGr()
		
	
			
	def lexm (self,graphC,EdgList):
		"""Lex M"""
		graphC = nx.Graph(graphC)
		#declared empty label dictionary and numbered vertices dictionary
		keys = self.vertexList
		self.LabelsDict = {k: [] for k in keys} #(self.vertexList, []) #l(v)=Null
		self.NumberDict = dict.fromkeys(self.vertexList, None)
		rankv = self.noNodes #ranks given to vertices
			
		r = (len(self.unnumberedVertices) -1)
		for v in range(r, 0, -1):
			S = []
			if  v == r: #first iteration to choose arbitrary last vertex and give it rank n
				#print "Starting V ",v
				
				self.LabelsDict[v]  = [] 
				self.NumberDict[v] = self.noNodes #alpha(v) = i

				self.fc = v # removing this vertex from unnumbered vertex list to numbered vertex list
				self.unnumberedVertices.remove(self.fc)
				self.numberedVertices.append(self.fc)


				#print "Vertex starting with",v

				#other iterations
			else:
				#print "UL in next iteration", self.unnumberedVertices
				#finding vertex v of lexicographically maximum label in every iteration
				mat = max((len(ve), k) for k,ve in self.LabelsDict.iteritems() if k in self.unnumberedVertices)[1:]

				v = mat[0]
				#print "Vertex Selected (v):",v


				rankv = rankv -1

			for u in self.unnumberedVertices:

				if u != v:
					#print "U is:",u

					dep = []
					dep.append(v)

					dep.append(u)
					nbc = list(itertools.combinations(dep,2))#list containg u,v as tuples p

						
					for p in nbc:
						
						v1 = p[0]
						v2 = p[1]
					#if there is an edge uv or path u,.....,v
					self.G = nx.Graph(self.G)
					if (self.G.has_edge(*p)) or (nx.has_path(self.G,v,u)):

						if self.G.has_edge(*p):
							#print "Already Edge is there",p
							S.append(u)

						if (nx.has_path(self.G,u,v)):

							paths = nx.all_simple_paths(self.G, source=u, target=v)#gives all simple paths between u and v
							dps = (list(paths))

							if dps:
								uval = self.LabelsDict.get(u)#u's label list

								ulex = len(list(filter(bool,uval)))#lexig length of label of u




								for plist in dps:
									g = len(plist)
									count = 0

									for x in range(1, g-1):
										
										xval = self.LabelsDict.get(plist[x])
										xlex = len(list(filter(bool,xval)))

										if (plist[x] in self.unnumberedVertices) and (xlex < ulex):
											count += 1

									if (count == (g-2)) and (count != 0) and ((g-2) != 0):

										S.append(u)

						#else:
							#print "No path between uv"


					

		
							
				else:
					#print "Consider another unnumbered vertex u"
					continue




				Sd = list(set(self.unnumberedVertices) & set(S)) #final S excluding numbered vertices

				#print "S to consider is Sd:",Sd

			#print "final s for:",v,"is:",S
			

			if v != r:
				self.unnumberedVertices.remove(v)

				self.numberedVertices.append(v)

			#Adding LAbels to Label Dictionary
			for vert in Sd:

				if rankv not in self.LabelsDict[vert]:
					self.LabelsDict[vert].append(rankv)

			
			#print self.LabelsDict
		

			for ut in Sd:

				#print "Start for fill edges:", Sd
				#print "Edges are:",v,"and",ut
				if (v != ut): #ut is unnnumbered vertex in S to avoid self loops

					#if self.G.has_edge(v,ut):
						#print "uv belongs to Edgelist:"

					if not (self.G.has_edge(v,ut)):
						self.Fill.append((v,ut))
						#print "Added Edge as uv does not belong to EdgeList:", v,ut



					#print "Fill edges added are", self.Fill
			#print "unn",self.unnumberedVertices
			#print "num",self.numberedVertices


						
			self.NumberDict[v] =  rankv#alpha(v) = i
			
	def FinalLGraph(self,graphVerify,newaddedgelist,vertexlist):
		print "EdgeList verifying",newaddedgelist
		print "Total Edges added in LexM is ",len(newaddedgelist)
		GD = nx.Graph(self.G)
		pos = nx.spring_layout(GD)

		B = copy.deepcopy(self.G)
		B = nx.Graph(B)
		B.add_nodes_from(vertexlist)
		B.add_edges_from(newaddedgelist)
		B = nx.to_dict_of_lists(B)
		print "see B", B
		##Recognition----
		graph = nx.Graph(B)
		print "We could"
		#print type(B)
		if nx.is_chordal(graph):
			print "IT IS CHORDAL"
		else :
			print "NO IT IS NOT CHORDAL"

		#print "Draw graph"
		nx.draw_networkx_nodes(GD, pos, nodelist=vertexlist, node_color='red', node_size=300, alpha=0.8,label='Min degree')	
		nx.draw_networkx_edges(GD, pos, width=1.0, alpha=0.5)
		nx.draw_networkx_edges(GD, pos, edgelist=newaddedgelist, width=8.0, alpha=0.5, edge_color='blue',label='Min degree')
		nx.draw_networkx_labels(GD,pos)
		plt.draw()
		plt.show()	

	def finalDisplay(self):

		if self.unnumberedVertices:
			#print "I am alive", self.unnumberedVertices
			self.numberedVertices = self.numberedVertices + self.unnumberedVertices
			#print "New", self.numberedVertices
		for m in self.unnumberedVertices:
			self.NumberDict[m] = 1
			#print "Heck", self.NumberDict
			self.unnumberedVertices.remove(m)
		#print "Displaying Vertex with Number",self.NumberDict
		for k,v in self.LabelsDict.iteritems():
			v.sort()
			self.LabelsDict[k] = [item for item, _ in itertools.groupby(v)]
		#print "Rem dup",self.LabelsDict
		#print "Final Label Dictionary:", self.LabelsDict
		#print "Number Dict",self.NumberDict
		#print "Numbered Vertex List", self.numberedVertices
		#print "Unnumbered Vertex List", self.unnumberedVertices
		#print "Final Label Dictionary:", self.LabelsDict
		#print "Edges added to make it chordal",self.Fill
		#print "Number of edges added:",len(self.Fill)
		#self.meo = self.numberedVertices[::-1]
		#print "Minimal Elimination Ordering is:",self.meo

val1 = int(raw_input("Enter no. of nodes:"))
val2 = int(raw_input("Enter no. of edges:"))
gvert = ChordalVert(val1,val2)
gvert.ArbitraryGraph()
#gvert.createChrdG()
#gvert.workLT()
#gvert.LChGr()


