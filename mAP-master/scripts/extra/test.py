import sys
import os
import glob
import xml.etree.ElementTree as ET

# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
py = os.path.dirname(__file__)
py2 =os.path.abspath(__file__)
py3= os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
py4= GT_PATH = os.path.join(parent_path, 'input','ground-truth')
print(py)
print(py2)
print(py3)
print(py4)