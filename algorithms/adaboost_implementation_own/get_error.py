import sys

correct_file = sys.argv[1]
predicted_file = sys.argv[2]

with open(correct_file) as f1, open(predicted_file) as f2:
	nr = 0
	
	for line1, line2 in zip(f1.readlines(), f2.readlines()):
		if line1 and line2:
			nr += (line1.split()[-1]) != (line2.split()[-1])
	
	print(nr)
