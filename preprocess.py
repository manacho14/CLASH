
import scipy.misc
import numpy as np
import pandas as pd
from astropy.io import fits
import cv2
from skimage.transform import resize

class Preprocess(object):

	def __init__(self, IDs, path):
		self.IDs = IDs
		self.path = path

	def get_labeled_data(self, data):
		X = self.ids_to_image()
		X = self.normalize(X)
		y = data.drop('ID', axis = 1).reset_index(drop=True)

		return X, y

	def get_images(self):
		X = self.ids_to_image()
		X = self.normalize(X)

		return X

	def ids_to_image(self):
		images = []
		for id_ in self.IDs:
			images.append(self.open_image(id_))

		return np.stack(images, axis = 0)

	def open_image(self, ID):
		image = fits.open(self.path + ID + '.fits')
		image = image[0].data
		image = scipy.misc.imresize(image.astype('float32'), (80, 80)) #cv2.resize(image.astype('float32'), dsize=(80,80), interpolation = cv2.INTER_NEAREST)

		return image.reshape(1,80,80)

	def normalize(self, data):
		mn = np.min(data)
		mx = np.max(data)

		return ((data - mn)/(mx - mn))





if __name__ == '__main__':

	dataset = 'CANDELS'
	data = pd.read_csv('{}/{}_labels.csv'.format(dataset,dataset), index_col = None)[:10]
	prep = Preprocess(data, '{}/stamps/'.format(dataset))

	X, y = prep.get_data()
