import os
from abc import ABC, ABCMeta
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToPILImage
import torch
from .Transforms import InputDataTransform

DATA_ROOT = '/home/ahmad/Project/dise/flask-dise/static/img/sample3'


class DesignDataset(VisionDataset, metaclass=ABCMeta):

	def __init__(self, root: str = DATA_ROOT, ) -> None:
		transform = InputDataTransform()

		super().__init__(root, transform=transform)

	def open_image(self, name):
		full_path = os.path.join(self.root, name)
		assert os.path.isfile(full_path)
		img = torchvision.io.read_image(full_path).type(dtype=torch.float32)

		return self.transform(img)

	def get_abs_path(self):
		return os.path.abspath(self.root)

	@property
	def data_list(self):
		abs_path = self.get_abs_path()
		res = os.listdir(abs_path)
		return res

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		el_name = self.data_list[idx]

		if type(el_name) is list:
			res = list(map(lambda x: self.open_image(x), el_name))
		else:
			res = self.open_image(el_name)

		return res

	@classmethod
	def get_images(cls, img_tensor):
		transform = ToPILImage()
		img = transform(img_tensor)
		return img


if __name__ == '__main__':
	from Dataset import DesignDataset

	data_set = DesignDataset(root_dir="/home/ahmad/Project/dise/flask-dise/static/img/DataSet/")
