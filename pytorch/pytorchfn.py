import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import (cos, sin, pi, rand, atan2, sum, manual_seed, ones, clone)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float

"""
Create Your Own Active Matter Simulation (With Python)
Philip Mocz (2021) Princeton Univeristy, @PMocz

Simulate Viscek model for flocking birds

"""

#Pytorch implementation
#Everything with np. is replaced with the torch.
#@profile
def active_matter(timesteps=200, birds=500):
	""" Finite Volume simulation """
	print("Running Pytorch implementation")
	
	# Simulation parameters
	v0           = 1.0      # velocity
	eta          = 0.5      # random fluctuation in angle (in radians)
	L            = 10       # size of box
	R            = 1        # interaction radius
	dt           = 0.2      # time step
	Nt           = timesteps      # number of time steps
	N            = birds      # number of birds
	plotRealTime = False
	
	# Initialize
	manual_seed(17)      # set the random number generator seed
 
	# Bird positions
	x = torch.rand(N, device=device) * L
	y = torch.rand(N, device=device) * L

	# Bird velocities
	theta = 2 * torch.pi * torch.rand(N, device=device)
	vx = v0 * torch.cos(theta)
	vy = v0 * torch.sin(theta)
	
	# Prep figure
	fig = plt.figure(figsize=(4,4), dpi=80)
	ax = plt.gca()
	
	# Simulation Main Loop
	for i in range(Nt):

		# move
		x += vx*dt
		y += vy*dt
		
		# apply periodic BCs
		x = x % L
		y = y % L
		
		# Here is the main difference between the original and the pytorch implementation
		# Where the loop is replaced with vectorized operations
		# Which makes the program easier to parallelize
		# Contributing to the performance improvement
		# Create a matrix of all pairs of positions
		distances = ((x.view(-1, 1) - x.view(1, -1))**2 + (y.view(-1, 1) - y.view(1, -1))**2).sqrt()

		# Find neighbors
		neighbors = distances < R

		# Compute mean_theta using vectorized operations
		sx = (torch.cos(theta) * neighbors.float()).sum(dim=1)
		sy = (torch.sin(theta) * neighbors.float()).sum(dim=1)
		mean_theta = torch.atan2(sy, sx)
			
		# add random perturbations
		theta = mean_theta + eta*(rand(N, device=device)-0.5)
		
		# update velocities
		vx = v0 * cos(theta)
		vy = v0 * sin(theta)
  
		# plot in real time
		if plotRealTime:
			plt.cla()
			plt.quiver(x.to('cpu').numpy(),y.to('cpu').numpy(),vx.to('cpu').numpy(),vy.to('cpu').numpy())
			ax.set(xlim=(0, L), ylim=(0, L))
			ax.set_aspect('equal')	
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.pause(0.001)
				
	# Save figure
	plt.savefig('activematter.png',dpi=240)
	plt.close()
	#plt.show()
	    
	return 0

