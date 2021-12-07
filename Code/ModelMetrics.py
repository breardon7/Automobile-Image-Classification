import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, cohen_kappa_score, \
    classification_report
from tensorflow.python.keras.utils.np_utils import to_categorical

# Hyper Parameters
IMAGE_SIZE = 224
CHANNELS = 3
test_images = np.load('DataStorage/test_data.npy', allow_pickle=True)
print(len(test_images))
x_test = np.array([i[0] for i in test_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
y_test = np.array([i[1] for i in test_images])
y_test = to_categorical(y_test)
y_test = np.argmax(y_test, axis=1)


def model_predict(trained_model):
    y_pred = trained_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))
    print("The Hamming loss is: " + str(hamming_loss(y_test, y_pred)))
    print("The Accuracy score is: " + str(accuracy_score(y_test, y_pred)))
    print("The Cohen Kappa score is: " + str(cohen_kappa_score(y_test, y_pred)))

    #matrix = confusion_matrix(y_test, y_pred)
    #display_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix)
    #display_matrix.plot()
    #plt.show()





