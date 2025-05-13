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
from system import SiameseNetwork


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])


model = SiameseNetwork().cuda()
model.load_state_dict(torch.load("iris/siamese_model.pth"))
model.eval()


def extract_embeddings(image,model,transform):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0)
        image = image.cuda()
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        embeddings = model.forward_one(image)
    return embeddings  


image_dic= {

}
file_path = "iris/users.txt"

with open(file_path,'r') as file:
    file_content = file.read()

lines = file_content.strip().split('\n')

for line in lines:
    path,user_info = line.split(' (User ID: ')
    user_id = int(user_info.strip(')'))
    image_dic[user_id] = path



registered_users = {}

for user_id, path in image_dic.items():
    img = Image.open(path).convert('L') 
    features = extract_embeddings(img,model,transform)
    registered_users[user_id] = features


with open("iris/reg_users2.txt" ,"wb") as file:
    pickle.dump(registered_users, file)  