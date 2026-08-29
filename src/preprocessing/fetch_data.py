import moabb
from moabb.datasets import BNCI20140_01
from moabb.paradigms import MotorImagery


dataset = BNCI2014_001()
dataset.download()
print("Dataset downloaded and ready for processing.")