import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import ListedColormap
from matplotlib import cm 
from matplotlib.patches import FancyArrow

class Game:
	def __init__(self, gt_rewards, batch, policy, learned_rewards, iteration, data_idx=None):
		self.shape = int(np.sqrt(gt_rewards.shape[0]))
		self.gt_rewards_ = gt_rewards.reshape(self.shape,self.shape)
		self.batch_ = batch
		self.policy_ = np.argmax(policy, 1).reshape(self.shape,self.shape)
		self.learned_rewards_ = learned_rewards
		self.teacher_idx_ = data_idx
		#print(self.teacher_idx_)
		self.selected_idx_ = None
		self.iteration_ = iteration 
	def __index_to_arrow(self, n, direction):
		size = self.shape
		r, c = n // size, n % size
		# n = 0
				# r, c = 0, 0
		x, y = c, self.shape - 1 -  r	
		if direction == 3: 
			dx, dy = 1, 0
			y += 0.5
			x += 0.1

		if direction == 2:
			y += 0.5
			x += 0.9
			dx, dy = -1, 0

		if direction == 1:
			dx, dy = 0, -1
			y += 0.9
			x += 0.5

		if direction == 0:
			x += 0.5
			y += 0.1
			dx, dy = 0, 1
		return x, y, dx * 0.5, dy * 0.5

	def __grid_to_arrow(self, r, c, direction):
		x, y = c, self.shape - 1 -  r

		if direction == 3: 
			dx, dy = 1, 0
			y += 0.5
			x += 0.1

		if direction == 2:
			y += 0.5
			x += 0.9
			dx, dy = -1, 0

		if direction == 1:
			dx, dy = 0, -1
			y += 0.9
			x += 0.5

		if direction == 0:
			x += 0.5
			y += 0.1
			dx, dy = 0, 1
		return x, y, dx * 0.5, dy * 0.5


	def display(self):
		seed = 0
		size = self.shape
		alphabet = "ABCDEFGHI"
		numbers = "123456789"

		gradient = cm.get_cmap('bwr', 100)
		cmap = ListedColormap(gradient(np.append(np.linspace(0, 0.5, 5), np.linspace(0.5, 0.9, 25))))

		#cmap = ListedColormap(gradient(np.linspace(0, 1, 15)))

		
		reward_min = np.min(self.gt_rewards_)
		reward_max = np.max(self.gt_rewards_)

		fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5), constrained_layout=True)

		for i in range(3):
			if i == 0:
				ax[i].imshow(self.gt_rewards_, interpolation='none', cmap=cmap, vmin=reward_min, vmax=reward_max, extent=[0, size, 0, size], zorder=0, picker=10)

			ax[i].set_xticklabels(list(alphabet))
			ax[i].set_yticklabels(list(numbers))
			ax[i].set_xticks(np.arange(0.5, size + 0.5, 1))
			ax[i].set_yticks(np.arange(0.5, size + 0.5, 1))

			for x in range(self.shape):
				ax[i].axhline(x, lw=1, color='k', zorder=self.shape)
				ax[i].axvline(x, lw=1, color='k', zorder=self.shape)

				ax[i].axhline(x, lw=1, color='k', zorder=self.shape)
				ax[i].axvline(x, lw=1, color='k', zorder=self.shape)

		# --- DRAW 1ST SUBPLOT 

		scale = 1

		for p in range(len(self.batch_[0])):
			x2, y2, dx2, dy2 = self.__index_to_arrow(self.batch_[0][p], self.batch_[1][p])
			arrow = ax[0].arrow(x2, y2, dx2 * scale, dy2 * scale, head_width=0.25, head_length=0.25, fc='k', ec='k', gid=p, picker=True)

			if self.teacher_idx_ is not None and p == self.teacher_idx_:
				self.correct_arrow_ = arrow

		ax[0].set_title('Ground Truth Reward Map')
		# --- DRAW 2ND SUBPLOT
		ax[1].imshow(self.learned_rewards_[0].reshape((self.shape,self.shape)), vmin=reward_min, vmax=reward_max, interpolation='none', cmap=cmap, extent=[0, size, 0, size], zorder=0)
		#print(self.learned_rewards_[0].reshape(self.shape, self.shape))
		#print(self.learned_rewards_)
		#test = np.asarray([[-2.2, -1.8, -1.6, -1.4, -1.2], [-1, -0.8, -0.6, -0.4, -0.2], [0, 0.5, 1, 1.5, 2], [2.5, 3, 3.5, 4, 4.5], [5, 5.5, 6,  6.5, 7]])
		# print(test)
		# ax[1].imshow(test, vmin=reward_min, vmax=reward_max, interpolation='none', cmap=cmap, extent=[0, size, 0, size], zorder=0)

		for p in range(len(self.batch_[0])):
			x2, y2, dx2, dy2 = self.__index_to_arrow(self.batch_[0][p], self.batch_[1][p])
			ax[1].arrow(x2, y2, dx2 * scale, dy2 * scale, head_width=0.25, head_length=0.25, fc='k', ec='k', gid=p, picker=True)

		ax[1].set_title('Learner Current Reward Map')

		# --- DRAW 3RD SUBPLOT
		empty = np.ones((self.shape,self.shape))
		cmap2 = ListedColormap(['white'])
		ax[2].imshow(empty, cmap=cmap2, interpolation='none', extent=[0, size, 0, size], zorder = 0)

		for r_ in range(self.shape):
			for c_ in range(self.shape):
				x, y, dx, dy = self.__grid_to_arrow(r_, c_, self.policy_[r_][c_])
				ax[2].arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')

		ax[2].set_title('Most Probable Learner Actions')
		fig.suptitle('Iteration %d' % (self.iteration_))
		# --- Add pick
		def close_all(event):
			plt.close('all')

		def onpick(event):
			artist = event.artist
			if isinstance(artist, FancyArrow):
				self.selected_idx_ = artist._gid
				print(self.selected_idx_)

				# # show correct choice
				if (self.teacher_idx_ is not None):
					self.correct_arrow_.set_color('green')
					fig.canvas.draw()
					plt.close('all')#fig.canvas.mpl_connect('pick_event', close_all)
				else:
					plt.close('all')
		fig.canvas.mpl_connect('pick_event', onpick)
		plt.show(block=True)
