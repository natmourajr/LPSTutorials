""" 
  This file contents ARTNet implementation
"""


import numpy as np

class TrnParams(object):
	def __init__(self, learning_rate=0.1,verbose= False):
		self.learning_rate = learning_rate
		self.verbose = verbose
	
	def Print(self):
		print 'Class ARTNet TrnParams'
		print 'Learning Rate: %1.5f'%(self.learning_rate)
		if self.verbose:
			print 'Verbose: True'
		else:
			print 'Verbose: False'

class ARTNet(object):

	"""
	ARTNet class
		This class implement the Adaptive Resonance Theory
	"""		
	def __init__(self, n_clusters=2, similarity_radius=0.1, dist="euclidean", randonize=True, dev=True):
		"""
		ARTNet constructor
			n_clusters: Number of cluster to be used (default: 2)
			similarity_radius: Similarity Radius (default: 0.1)
			dist: distance method used (defaults: euclidean)
			randonize: do or not random access to data            
			dev: Development flag
		"""
		self.n_clusters = n_clusters
		self.clusters = None
		self.cluster_last_used = None
		self.similarity_radius = similarity_radius
		self.dist = dist
		self.randonize = randonize
		self.dev = dev
        
	def calc_dist(self,pt1, pt2):
		if self.dist == "euclidean":
			return np.linalg.norm((pt1-pt2),ord=2)
        
	def create_cluster(self,new_cluster):
		if self.clusters is None:
			self.clusters = new_cluster
		else:
			self.clusters = np.append(self.clusters,new_cluster,axis=1)
        
	def update_cluster(self, cluster_id, event):
		self.clusters[:,cluster_id] = (self.clusters[:,cluster_id] + 
                                       self.learning_rate*
                                       (self.clusters[:,cluster_id]-event))

	def fit(self, data, trn_params=None):
		if trn_params is None:
			trn_params = TrnParams()
            
		if self.dev:
			trn_params.Print()
            
		if self.randonize:
			if data.shape[0] < data.shape[1]:
				trn_data = data[:,np.random.permutation(data.shape[1])].T
			else:
				trn_data = data[np.random.permutation(data.shape[0]),:]
		else:
			if data.shape[0] < data.shape[1]:
				trn_data = data.T
			else:
				trn_data = data
                
		for ievent in range(trn_data.shape[0]):
			if self.clusters is None:
				self.create_cluster(trn_data[ievent,:])
			else:
				# how to create a new cluster?!
				if len(self.clusters.shape) == 1:
					# compare with the first cluster
					print trn_data[ievent,:]
					print self.clusters
					dist = self.calc_dist(trn_data[ievent,:],self.clusters)
					if dist > self.similarity_radius:
						self.create_cluster(trn_data[ievent,:])
					else:
						self.update_cluster(0,trn_data[ievent,:])
				else:
					mat_dist = np.zeros([])
					for icluster in range(self.clusters.shape[1]):
						mat_dist[icluster] = self.calc_dist(trn_data[ievent,:],self.clusters[icluster,:])
					if np.min(mat_dist) > self.similarity_radius:
						self.create_cluster(trn_data[ievent,:])
                        

                