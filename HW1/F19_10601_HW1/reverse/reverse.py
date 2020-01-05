import sys

def main():
	num_args = len(sys.argv)
	print("Number of commandline args passed: %s" % num_args)
	if num_args < 3:
		print("Please give all the commandline args in the order: filename, input_filename, output_filename respectively. Current args given: %s" % sys.argv)
	with open(sys.argv[1]) as f:
		data = f.read()
	
	print("d: %s" % d)
	d.reverse()
	if d:
		d = d[1:]
		p = '\n'.join(d)
	with open(sys.argv[2], 'w') as f:
                f.write(d)
	print("Task done")


if __name__ == "__main__":
	main()
