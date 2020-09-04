import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import ListedColormap
from matplotlib import cm 
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrow
import time 

class Game:
	def __init__(self, gt_rewards, batch, policy, learned_rewards, iter_num, total_iters, data_idx=None):
		self.shape = int(np.sqrt(gt_rewards.shape[0]))
		self.gt_rewards_ = gt_rewards.reshape(self.shape,self.shape)
		self.batch_ = batch
		self.policy_ = np.argmax(policy, 1).reshape(self.shape,self.shape)
		self.learned_rewards_ = learned_rewards
		self.teacher_idx_ = data_idx
		self.iter_ = iter_num
		self.total_iters_ = total_iters 
		self.selected_idx_ = None
		self.arrows0 = []
		self.arrows1 = []

	def __index_to_arrow(self, n, direction):
		size = self.shape
		r, c = n // size, n % size
		# n = 0
				# r, c = 0, 0
		x, y = c, self.shape - 1 -  r
		rx = c
		ry = y	
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
		return x, y, dx * 0.5, dy * 0.5, rx, ry

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

		fig, ax = plt.subplots(1, 3, figsize=(7.9, 3))#, constrained_layout=True)
		text = fig.text(0.4,0.1, "")

		ax[0].set_position([0.05, 0.20399999999999999, 0.2, 0.7])
		ax[1].set_position([0.4, 0.20399999999999999, 0.2, 0.7])
		ax[2].set_position([0.729, 0.20399999999999999, 0.2, 0.7])
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
			x2, y2, dx2, dy2, rx, ry = self.__index_to_arrow(self.batch_[0][p], self.batch_[1][p])
			self.arrows0.append(ax[0].arrow(x2, y2, dx2 * scale, dy2 * scale, head_width=0.25, head_length=0.25, fc='k', ec='k', gid=p, picker=True))

			rect = Rectangle((rx, ry), 1, 1, fill=False, linewidth = 0.1, alpha=1, gid=p, picker=True)
			ax[0].add_patch(rect)

		ax[0].set_title('Ground Truth Reward Map')
		# --- DRAW 2ND SUBPLOT
		ax[1].imshow(self.learned_rewards_[0].reshape((self.shape,self.shape)), vmin=reward_min, vmax=reward_max, interpolation='none', cmap=cmap, extent=[0, size, 0, size], zorder=0)

		for p in range(len(self.batch_[0])):
			x2, y2, dx2, dy2, rx, ry = self.__index_to_arrow(self.batch_[0][p], self.batch_[1][p])
			self.arrows1.append(ax[1].arrow(x2, y2, dx2 * scale, dy2 * scale, head_width=0.25, head_length=0.25, fc='k', ec='k', gid=p, picker=True))

			rect = Rectangle((rx, ry), 1, 1, fill=False, linewidth = 0.1, alpha=1, gid=p, picker=True)
			ax[1].add_patch(rect)

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

		fig.suptitle("Training Iteration %d of %d" % (self.iter_, self.total_iters_), size='large')

		# --- Add pick
		def close_all(event):
			if event.key == 'y' or event.key == 'c':
				self.arrows0[self.teacher_idx_].set_color('y')
				self.arrows1[self.teacher_idx_].set_color('y')
				
				plt.close('all')

		def show_teacher(event):
			if (self.selected_idx_ is not None) and (event.key == 'y' or event.key == 'c'):
				if (self.teacher_idx_ is not None):
					self.arrows0[self.teacher_idx_].set_color('saddlebrown')
					self.arrows1[self.teacher_idx_].set_color('saddlebrown')
				text.set_position((0.25,0.1))
				text.set_text('Click the "‣" button in the menubar to run the text iteration')
				fig.canvas.draw()
				#plt.close('all')
		def onpick(event):
			text.set_position((0.31,0.1))
			text.set_text('Press "c" to confirm the selected arrow')
			artist = event.artist
			if isinstance(artist, FancyArrow) or isinstance(artist, Rectangle):
				print(artist._gid)

				if (self.selected_idx_ is not None):
					self.arrows0[self.selected_idx_].set_color('black')
					self.arrows1[self.selected_idx_].set_color('black')
				if (self.teacher_idx_ is not None):
					self.arrows0[self.teacher_idx_].set_color('black')
					self.arrows1[self.teacher_idx_].set_color('black')
				self.selected_idx_ = artist._gid

				self.arrows0[self.selected_idx_].set_color('green')
				self.arrows1[self.selected_idx_].set_color('green')
				
				
				# show correct choice
				'''if (self.teacher_idx_ is not None): 
					fig.canvas.mpl_connect('key_press_event', show_teacher)
				'''
				fig.canvas.draw()
		fig.canvas.mpl_connect('key_press_event', show_teacher)

		fig.canvas.mpl_connect('pick_event', onpick)
		plt.show()