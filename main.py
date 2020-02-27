import sys
from coil100 import trainCoil100
from cifar10 import trainCifar10
from mnist import trainMnist
from fashion import trainFashion

def error(aa):
	if (len(aa) !=3):
		print("uso: main.py 1(Coil100)/2(Cifar)/3(Fasion)/4(Mnist) 18/32/50")

#print( "This is the name of the script: ", sys.argv[0])
#print( "Number of arguments: ", len(sys.argv))
#print( "The arguments are: " , str(sys.argv))

print("Executing... ", sys.argv[0], str(sys.argv), "...")

for i in range(len(sys.argv)):
	print(sys.argv[i])

if len(sys.argv) != 3: error(sys.argv)


if int(sys.argv[1]) == 1: #Coil
	if int(sys.argv[2]) == 18 or int(sys.argv[2]) == 34 or int(sys.argv[2]) == 50:
		print("coil100 ")
		trainCoil100(sys.argv[2])

if int(sys.argv[1]) == 2: #Cifar10
	if int(sys.argv[2]) == 18 or int(sys.argv[2]) == 34 or int(sys.argv[2]) == 50:
		print('trainCifar10')
		trainCifar10(sys.argv[2])

if int(sys.argv[1]) == 3: #Fashion
	if int(sys.argv[2]) == 18 or int(sys.argv[2]) == 34 or int(sys.argv[2]) == 50:
		print('trainFashion')
		trainFashion(sys.argv[2])

if int(sys.argv[1]) == 4: #Mnist
	if int(sys.argv[2]) == 18 or int(sys.argv[2]) == 34 or int(sys.argv[2]) == 50:
		print('trainMnist')
		trainMnist(sys.argv[2])

