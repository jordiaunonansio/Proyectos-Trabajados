from model_CRNN_seq import CRNN, SimpleCNN,  val_loader
import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torchvision import datasets, transforms
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

abc = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25
}

# Create CNN model
cnn_model = SimpleCNN()
# Load the state dictionary
state_dict = torch.load('/home/xnmaster/XNAPproject-grup_08/model_CNN/results/dataset2/trained_model.pth', map_location=torch.device('cpu'))

# Create the CRNN model
crnn_model = CRNN(cnn_model, rnn_hidden_size=256, num_classes=27)  # 26 letters + 1 for blank

# Load the state dictionary
state_dict = torch.load('/home/xnmaster/XNAPproject-grup_08/model_CRNN/crnn_model.pth', map_location=torch.device('cpu'))



# Load the state dictionary
crnn_model.load_state_dict(state_dict)
# Set the model to evaluation mode
crnn_model.eval()


# Define the loss function and the optimizer
loss_function = CTCLoss(blank=26, zero_infinity=True)  # Assuming blank label is 26
optimizer = optim.Adam(crnn_model.parameters())


# testing loop

lib_images = []
lib_labels = []
lib_lengths = []

word_images = []
word_labels = []
word_lengths = []
lns = []
for i, (images, labels, lengths) in enumerate(val_loader):
    lns.append(lengths)
    # If the current word is finished, process the batch and start a new one
    if i > 0 and lns[i-1] != lns[i]:
        word_images = torch.stack(word_images)
        word_labels = torch.stack(word_labels)
        word_lengths = torch.stack(word_lengths)
        # Move the images and labels to the GPU if available
        lib_images.append(word_images)
        lib_labels.append(word_labels) 
        lib_lengths.append(word_lengths)
        
        word_images = []
        word_labels = []
        word_lengths = []
    word_images.append(images)
    word_labels.append(labels)
    word_lengths.append(lengths)
test_loss = 0
correct = 0
    # process the batch
lib_images = lib_images[1:]
lib_labels = lib_labels[1:]
lib_lengths = lib_lengths[1:]
with torch.no_grad():
    for word_images, word_labels, word_lengths in zip(lib_images, lib_labels, lib_lengths):
        word_images = word_images.to(device)
        word_labels = word_labels.to(device)
        word_lengths = word_lengths.to(device)
        outputs = crnn_model(word_images)
        outputs = outputs.transpose(0, 1)
        input_lengths = torch.tensor([outputs.size(0)], dtype=torch.long)        
        target_lengths = torch.tensor([word_labels.size(0)], dtype=torch.long)
        word_labels = word_labels.unsqueeze(0)
        word_labels = word_labels.view(-1)
        loss = loss_function(outputs, word_labels, input_lengths, target_lengths)
        test_loss += loss.item()
        pred = outputs.argmax(dim=2)
        pred_str = ''
        for l in pred:
            pred_str += inverted[l.item()]
        if pred_str == word_labels:
            correct += 1
test_loss /= len(val_loader)
accuracy = correct / len(val_loader)
print(f'Loss: {test_loss}, Accuracy: {accuracy}')


# Save the model
torch.save(crnn_model.state_dict(), 'crnn_model.pth')


