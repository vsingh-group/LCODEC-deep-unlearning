from torch import nn
from utils import GradientReversal

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.feature_extractor = nn.Sequential(
			nn.Conv2d(3, 20, kernel_size=3),
			# nn.MaxPool2d(2),
			# nn.ReLU(),
			nn.Conv2d(20, 30, kernel_size=3),
			# nn.MaxPool2d(2),
			# nn.ReLU(),
			nn.Conv2d(30, 40, kernel_size=3),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Conv2d(40, 50, kernel_size=3),
			nn.MaxPool2d(2),
			# nn.Dropout2d(),
		)
		
		self.classifier = nn.Sequential(
			nn.Linear(8450, 1000),
			nn.ReLU(),
			nn.Linear(1000, 100),
			nn.ReLU(),
			# nn.Dropout(),
			nn.Linear(100, 2),
		)

	def forward(self, x):
		features = self.feature_extractor(x)
		features = features.view(x.shape[0], -1)
		logits = self.classifier(features)
		return logits


class DisCrim(nn.Module):
	def __init__(self, grl_lambda=0.1):
		super().__init__()
		self.discriminator = nn.Sequential(
	        GradientReversal(lambda_=grl_lambda),
	        nn.Linear(1280, 640),
			nn.ReLU(),
			nn.Linear(640, 100),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(100, 1),
	    )

	def forward(self, x):
		preds = self.discriminator(x)
		return preds

class New_DisCrim(nn.Module):
	def __init__(self, grl_lambda=0.1):
		super().__init__()
		self.discriminator = nn.Sequential(
	        GradientReversal(lambda_=grl_lambda),
			nn.Dropout(0.2),
			nn.Linear(1280, 2),
	    )

	def forward(self, x):
		preds = self.discriminator(x)
		return preds