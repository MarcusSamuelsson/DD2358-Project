import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport atan2, cos, sin, pow
from libc.stdlib cimport rand, RAND_MAX
# Thanks to @tryptofame for proposing an updated snippet

"""
Create Your Own Active Matter Simulation (With Python)
Philip Mocz (2021) Princeton Univeristy, @PMocz

Simulate Viscek model for flocking birds

"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(True)
def active_matter(int timesteps=200, int birds=500):
	""" Finite Volume simulation """
	print("Running Cython implementation")

	# This is the first part of the optimization, 
	# we declare the types of the variables
	# Variable declarations
	cdef double v0, eta, L, R, dt, dist, sx, sy
	cdef int Nt, N, i, b
	cdef double [:] x, y, theta, vx, vy, mean_theta


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
	np.random.seed(17)      # set the random number generator seed

	# bird positions
	x = np.random.rand(N)*L
	y = np.random.rand(N)*L

	# bird velocities
	theta = 2 * np.pi * np.random.rand(N)
	vx = v0 * np.cos(theta)
	vy = v0 * np.sin(theta)

	# Prep figure
	#fig = plt.figure(figsize=(4,4), dpi=80)
	ax = plt.gca()
	
	# Simulation Main Loop
	for i in range(Nt):

		
		# This is the second part of the optimization,
		# where we use the cythonized code
		# Loop over birds
		for b in range(N):
			# move
			x[b] += vx[b]*dt
			y[b] += vy[b]*dt

			# apply periodic BCs
			x[b] = x[b] % L
			y[b] = y[b] % L
		
		# This is the third part of the optimization,
		# where we use C functions for the trigonometric functions
		# to reduce the overhead of the Python interpreter
		# This is the most important part of the optimization
		# find mean angle of neighbors within R
		mean_theta = theta
		for b in range(N):
			sx = 0.0
			sy = 0.0
			for j in range(N):
				dist = pow(x[j] - x[b], 2) + pow(y[j] - y[b], 2)
				
				if dist < R**2:
					sx += cos(theta[j])
					sy += sin(theta[j])

			mean_theta[b] = atan2(sy, sx)
			
		
		
		# update velocities
		for b in range(N):
			# add random perturbations
			theta[b] = mean_theta[b] + eta*((rand()/RAND_MAX) - 0.5)

			vx[b] = v0 * cos(theta[b])
			vy[b] = v0 * sin(theta[b])
		
		# plot in real time
		if plotRealTime:
			plt.cla()
			plt.quiver(x,y,vx,vy)
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

