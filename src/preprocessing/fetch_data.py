import moabb
from moabb.datasets import BNCI20140_01
from moabb.paradigms import MotorImagery


dataset = BNCI20140_01()
dataset.download()
print("Dataset downloaded and ready for processing.")