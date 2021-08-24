import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

emotions_labels = ['Anger', 'Disgust ',' Fear ',' Joy ',' Sadness ',' Surprise ',' Indifference']
emotion_data = []
fer_path = 'dataset/fer2013.csv'
data = pd.read_csv(fer_path)
emotion_data = data['emotion'].value_counts().sort_index()
y_pos = np.arange(len(emotions_labels))
plt.bar(y_pos, emotion_data, align='center', alpha=0.5)
plt.xticks(y_pos, emotions_labels, rotation='vertical')
plt.ylabel('Number')
plt.xlabel('Emotions')
plt.title('The number of patterns in the FER2013 Dataset')
plt.show()

'''
epoches = []
acc = []
loss = []
data = np.genfromtxt('results.csv', delimiter=',', names=['epoches', 'acc', 'loss'])
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data['epoches']+1, data['acc'], color='r')
plt.xlabel('Iterate')
plt.ylabel('Precision')
plt.title('Model - Precision')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data['epoches']+1, data['loss'], color='r')
plt.xlabel('Iterate')
plt.ylabel('Losses')
plt.title('Model - Losses')
plt.show()

'''