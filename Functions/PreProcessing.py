""" 
  This file contents some processing functions
"""


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.layers import Merge

class TrnParams(object):
	def __init__(self, learning_rate=0.01, 
	learning_decay=1e-6, momentum=0.3, 
	nesterov=True, train_verbose=False, 
	n_epochs=500,batch_size=8):
		self.learning_rate = learning_rate
		self.learning_decay = learning_decay
		self.momentum = momentum
		self.nesterov = nesterov
		self.train_verbose = train_verbose
		self.n_epochs = n_epochs
		self.batch_size = batch_size
	
	def Print(self):
		print 'Class TrnParams'
		print 'Learning Rate: %1.5f'%(self.learning_rate)
		print 'Learning Decay: %1.5f'%(self.learning_decay)
		print 'Momentum: %1.5f'%(self.momentum)
		if self.nesterov:
			print 'Nesterov: True'
		else:
			print 'Nesterov: False'
		if self.train_verbose:
			print 'Train Verbose: True'
		else:
			print 'Train Verbose: False'
		print 'NEpochs: %i'%(self.n_epochs)
		print 'Batch Size: %i'%(self.batch_size)	
	

class PCDIndependent(object):
	def __init__(self, n_components=2):
		self.n_components = n_components
		self.models = {}
		self.trn_descs = {}
		self.pcds = None
		
	def fit(self, data, target, train_ids, test_ids, trn_params=None):
		
		if trn_params is None:
			trn_params = TrnParams()
		
		print 'Train Parameters'	
		trn_params.Print()
		
		for ipcd in range(self.n_components):
			if ipcd == 0:
				my_model = Sequential()
				
				# add a linear layer to isolate the input of NN model
				my_model.add(Dense(data.shape[1],input_dim=data.shape[1], init='identity',trainable=False))
				
				my_model.add(Activation('linear'))
				
				# add a non-linear single neuron layer to compress all information
				my_model.add(Dense(1, input_dim=data.shape[1], init='uniform'))
				my_model.add(Activation('tanh'))
				
				# add a non-linear output layer with max sparse target shape
				my_model.add(Dense(target.shape[1], init='uniform')) 
				my_model.add(Activation('tanh'))
				
				# creating a optimization function using steepest gradient
				sgd = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay, momentum=trn_params.momentum, nesterov=trn_params.nesterov)
				
				# compile the model
				my_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy','mean_squared_error'])
				
				# early stopping to avoid overtraining
				earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
				
				# Train model
				init_trn_desc = model.fit(data[train_ids], target[train_ids],nb_epoch=trn_params.n_epochs, batch_size=trn_params.batch_size,callbacks=[earlyStopping], verbose=trn_params.train_verbose,validation_data=(data[test_ids],target[test_ids]),shuffle=True)
				self.models[ipcd] = my_model
				self.trn_descs[ipcd] = init_trn_desc
				
				if self.pcds is None:
					self.pcds = model.layers[2].get_weights()[0]
				else:
					self.pcds = np.append(self.pcds,model.layers[2].get_weights()[0])
			else:
				print 'else'
		
		
		
		return self