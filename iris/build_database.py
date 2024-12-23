
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from PIL import Image
import random
import os
import cv2
import pickle


def build_database():
    registered_users = {
            "S5963L05": "G:\\4th year biomedical\\Biometrics\\code\\CASIA-Iris-Thousand\\CASIA-Iris-Thousand\\963\\L\\S5963L05.jpg",
    }
    with open("G:\\4th year biomedical\\Biometrics\\final\\iris\\reg_users.txt", "wb") as file:
          pickle.dump(registered_users, file)


build_database()