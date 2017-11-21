import matplotlib.pyplot as plt
import numpy as np
import os

n_r = n_c = 100

data_path = os.path.join("..", "__randu")
file_list = os.listdir(data_path)

for f in file_list:
	if "txt" not in f:	continue
	random_data = np.genfromtxt(os.path.join(data_path, f))
	random_data = np.reshape(random_data, (n_r, n_c))

	plt.matshow(random_data)
	plt.show()
	break

"""prng_data = np.random.randint(0, 2, (n_r, n_c))

plt.matshow(prng_data)
plt.show()"""