# 必要なものをインポート
import chainer
import chainer.functions as F
import chainer.links as L


####################################
######  Neural Network Class  ######
####################################
class NN(chainer.Chain):
	def __init__(self):
		super(NN, self).__init__(
			# Linear = 全結合層
			l1 = L.Linear(28*28, 100),
			l2 = L.Linear(100, 100),
			l3 = L.Linear(100,10)
		)

	def __call__(self, x, t):
		# 誤差の計算方法は 交差エントロピー誤差
		return F.softmax_cross_entropy(self.fwd(x), t)

	def fwd(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		return self.l3(h2)


####################################
# Convolution Neural Network Class #
####################################
class CNN(chainer.Chain):
	def __init__(self):
		super(CNN, self).__init__(
			# Convolution2D = ２次元の畳み込み層
			cn1 = L.Convolution2D(1, 20, 5),  # 28x28 -> 24x24
			cn2 = L.Convolution2D(20, 50, 5), # 12x12 -> 8x8
			l1 = L.Linear(800, 100),					# (4 x 4) x 5 = 800
			l2 = L.Linear(100,10)
		)

	def __call__(self, x, t):
		# 誤差の計算方法は 交差エントロピー誤差
		return F.softmax_cross_entropy(self.fwd(x), t)

	def fwd(self, x):
		h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)  # 24x24 -> 12x12
		h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2) # 8x8 -> 4x4
		h3 = F.dropout(F.relu(self.l1(h2)))
		return self.l2(h3)



