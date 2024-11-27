import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def plot_letter_accuracy(all_preds, all_labels, abc):
    #plot letter accuracy descending values
    letter_accuracy = {}
    for letter in abc.keys():
        letter_accuracy[letter] = 0
    for i in range(len(all_preds)):
        if all_preds[i] == all_labels[i]:
            letter_accuracy[list(abc.keys())[list(abc.values()).index(all_preds[i])]] += 1
    letter_accuracy = {k: v / (len(all_labels)/26) for k, v in letter_accuracy.items()}
    letter_accuracy = dict(sorted(letter_accuracy.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(10, 5))
    plt.bar(letter_accuracy.keys(), letter_accuracy.values())
    plt.xlabel('Letter')
    plt.ylabel('Accuracy')
    plt.title('Letter Accuracy')
    plt.grid()
    plt.savefig('model_CNN/results/dataset2/letter_accuracy.png')

def plot_accuracy_per_epoch(accuracy_per_epoch, traintest):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracy_per_epoch) + 1), accuracy_per_epoch, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'model_CNN/results/dataset2/accuracy_per_epoch_{traintest}.png')

def plot_loss_per_epoch(loss_per_epoch, traintest):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_per_epoch) + 1), loss_per_epoch, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'model_CNN/results/dataset2/loss_per_epoch_{traintest}.png')

def plot_confusion_matrix(all_preds, all_labels, abc):
    cm = confusion_matrix(all_labels, all_preds, labels=list(abc.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(abc.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('model_CNN/results/dataset2/confusion_matrix2.png')

def save_train_test_accuracy_plot(train_accuracy_per_epoch, test_accuracy_per_epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracy_per_epoch) + 1), train_accuracy_per_epoch, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracy_per_epoch) + 1), test_accuracy_per_epoch, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy per Epoch')
    plt.legend()
    plt.grid()
    plt.savefig('model_CNN/results/dataset2/train_test_accuracy_per_epoch.png')

def save_train_test_loss_plot(train_loss_per_epoch, test_loss_per_epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss_per_epoch) + 1), train_loss_per_epoch, label='Train Loss')
    plt.plot(range(1, len(test_loss_per_epoch) + 1), test_loss_per_epoch, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.savefig('model_CNN/results/dataset2/train_test_loss_per_epoch.png')