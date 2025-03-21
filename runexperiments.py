import os

print("C2: Time Measurement\n")
os.system("python lab2.py --path ./cifar --cuda")

print("C3: I/O Optimization\n")
os.system("python lab2.py --path ./cifar --cuda --c3")

print("C4: 1 worker vs 4\n")
print("1 Worker:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 1")
print("4 Workers:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4")

print("C5: GPU vs CPU\n")
print("GPU:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4")
print("CPU:\n")
os.system("python lab2.py --path ./cifar --loadworkers 4")

print("C6: Optimizers\n")
print("SGD:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4")
print("SGD with nesterov:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4 --optimizer sgdwn")
print("Adagrad:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4 --optimizer adagrad")
print("Adadelta:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4 --optimizer adadelta")
print("Adam:\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4 --optimizer adam")

print("C7: Remove batch norm\n")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4 --removeBN")

print("Q3: #Parameters and gradients with SGD")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4 --q3")

print("Q4: #Parameters and gradients with Adam")
os.system("python lab2.py --path ./cifar --cuda --loadworkers 4 --q3 --optimizer adam")
