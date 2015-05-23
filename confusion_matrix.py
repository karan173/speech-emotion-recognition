
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


target_names = ['anger', 'boredom', 'disgust', 'anxiety', 'happiness', 'sadness', 'neutral']

predictedY =  utteranceClf.predict(newTestX[:, 0:-1])
cm = confusion_matrix(newTestY, predictedY)
print(cm)
plt.figure()
plot_confusion_matrix(cm,target_names)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, target_names, title='Normalized confusion matrix')

plt.show()
