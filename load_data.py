from __future__ import print_function, division
import os
import glob
import copy
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class truly_random_dataset(Dataset):

	data = None
	n_rows = None
	transform = None

	def __init__(self, dataroot, n_rows, transform = None):
		for f in glob.glob(os.path.join(dataroot, "*.txt")):
			_data = np.loadtxt(f)

			if self.data is None:	self.data = copy.copy(_data)
			else:					self.data = np.append(self.data, _data)

		self.n_rows = n_rows
		self.transform = transform

	def __len__(self):
		return len(self.data) // self.n_rows ** 2

	def __getitem__(self, idx):
		data_idx = self.data[idx * self.n_rows ** 2:(idx + 1) \
		* self.n_rows ** 2].reshape(self.n_rows, -1)[None, :]

		#pdb.set_trace()
		
		if self.transform is not None:
			data_idx = self.transform(data_idx)
		
		return data_idx

class rescale(object):
	
	def __call__(self, sample):
		np.place(sample, sample == 0., -1.)
		return sample

class to_tensor(object):

	def __call__(self, sample):
		return (torch.from_numpy(sample).type(torch.FloatTensor))


