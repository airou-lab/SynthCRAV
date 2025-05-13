#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Neural Net plots
def plot_trainval_loss(hist_filename):
    plt.figure(figsize=(16, 9))

    # load hist
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f) 

    train_loss = hist['train_loss']
    val_loss = hist['val_loss']
    n_epochs = len(train_loss)

    print()
    print('n_epochs:',n_epochs)
    print('train_loss:',train_loss)
    print('val_loss:',val_loss)

    plt.plot(range(n_epochs),train_loss)
    plt.plot(range(n_epochs),val_loss)

    plt.title('Train/Val loss vs Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend(['Train loss','Val loss'])

    plt.show()



def plot_trainval_acc(hist_filename):
    plt.figure(figsize=(16, 9))

    # load hist
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f) 

    train_acc = hist['train_accuracy']
    val_acc = hist['val_accuracy']
    n_epochs = len(train_acc)

    print()
    print('n_epochs:',n_epochs)
    print('train_acc:',train_acc)
    print('val_acc:',val_acc)

    plt.plot(range(n_epochs),train_acc)
    plt.plot(range(n_epochs),val_acc)

    plt.title('Train/Val accuracy vs Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend(['Train accuracy','Val accuracy'])

    plt.show()


def get_test_acc(hist_filename):
    # load hist
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f) 

    test_loss = hist['test_loss']
    test_accuracy = hist['test_accuracy']

    print()
    print('test_loss:',test_loss)
    print('test_accuracy:',test_accuracy)

def get_TP_FP(hist_filename):
    with open(hist_filename, "rb") as f:
        hist = pickle.load(f)

    preds = hist['test_results']['preds']
    labels = hist['test_results']['labels']
    n_classes = 11
    TP = np.zeros(n_classes, dtype=int)
    FN = np.zeros(n_classes, dtype=int)

    for c in range(n_classes):
        TP[c] = np.sum((preds == c) & (labels == c))  # Correctly predicted class c
        FN[c] = np.sum((preds != c) & (labels == c))  # Missed class c

    print('TP:',TP,sum(TP))
    print('FN:',FN,sum(FN))

def plot_confusion_mat(y_true, y_pred, name):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    num_labels = np.arange(0,11,dtype=int).tolist()
    label_list = np.array([0,10,20,30,40,50,60,70,80,90,100])

    fig, ax = plt.subplots(figsize=(16,9))
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, labels=num_labels, display_labels=label_list, ax=ax, colorbar=False)
    disp.plot(ax=ax,cmap=plt.cm.Blues, xticks_rotation=45)
    plt.savefig(name,dpi=300)
    # plt.show()





# main
if __name__ == '__main__':
    print(50*'-','Loading RADAR results',50*'-')
    path = './ckpt/radar_model_hist.pkl'
    plot_trainval_loss(path)
    plot_trainval_acc(path)
    get_test_acc(path)
    get_TP_FP(path)    

    # input('press any key for cam model')
    print(50*'-','Loading CAMERA results',50*'-')
    path = './ckpt/camera_model_hist.pkl'
    plot_trainval_loss(path)
    plot_trainval_acc(path)
    get_test_acc(path)

    get_TP_FP(path)    




