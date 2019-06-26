from random import uniform

def generate_data(filename, m):
	with open(filename, 'w') as out:
		for _ in range(m):
			x1, x2 = uniform(0, 100), uniform(0, 50)
			# label = 1 if (x1 ** 2 + x2 ** 2 >= 75) else -1
			label = 1 if ((x1 <= 50 and x2 <= 20) or (x1 > 60 and x2 > 30)) else -1
			# label = -1 if (x1 <= 20 or x1 >= 60) else 1
			out.write('{} {} {}\n'.format(x1, x2, label))

generate_data('../datasets/corner/train.txt', 20000)
generate_data('../datasets/corner/validation.txt', 1000)
