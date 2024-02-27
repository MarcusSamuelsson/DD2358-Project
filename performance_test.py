import time
import cython
import sys

sys.path.insert(0, './normal')
sys.path.insert(1, './cython')
sys.path.insert(2, './pytorch')

from activematter import active_matter as active_matter_normal
from cythonfn import active_matter as active_matter_cython
from pytorchfn import active_matter as active_matter_pytorch

# Print the results to a file easily pasted into excel
def print_for_excell_to_file(filename, average_time, standard_deviation):
	""" Print the results to a file """
	with open(f'./data/{filename}.txt', "w") as file:
		file.write("Average time:\n")
		for time in average_time:
			file.write(f"{time}\n")

		file.write("\n")
		file.write("Standard deviation:\n")
		for var in standard_deviation:
			file.write(f"{var}\n")
		
	
	return 0
	


# Calculate the standard deviation of the data
def calculate_standard_deviation(data):
	""" Calculate the standard deviation of the data """
	n = len(data)
	mean = sum(data) / n
	variance = sum([((x - mean) ** 2) for x in data]) / n
	res = variance ** 0.5
	return res

# Run the function with different number of birds no itterations
def run_function(fn):
	""" Run the function """
	num_birds_variation = [100, 200, 300, 400, 500]
	
	for birds in num_birds_variation:
 		fn(birds=birds)

# Test the performance of the finite volume simulation
def performance_test_active_matter(fn):
	""" Test the performance of the finite volume simulation """
	itterations = 10
	num_birds_variation = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
	time_taken = []
	average_time = []
	standard_deviation = []

	# Run the function with different number of birds
	for birds in num_birds_variation:
		print(f"Testing with {birds} birds")
		# Run the function itterations times
		for i in range(itterations):
			t0 = time.perf_counter()
			fn(birds=birds)
			time_taken.append(time.perf_counter() - t0)
		
		# Calculate the average time and standard deviation
		avg = sum(time_taken) / itterations
		var = calculate_standard_deviation(time_taken)
		average_time.append(avg)
		standard_deviation.append(var)
		
		time_taken.clear()
	
	return average_time, standard_deviation

if __name__== "__main__":
	fn = [active_matter_normal, active_matter_cython, active_matter_pytorch]
	fn_name = ["normal","cython", "pytorch"]

	
	for f in fn:
		name = fn_name.pop(0)
		#run_function(f)
		average_time, standard_deviation = performance_test_active_matter(f)
		print_for_excell_to_file(name, average_time, standard_deviation)
		print(f"{name} done")
	
	print("All done")