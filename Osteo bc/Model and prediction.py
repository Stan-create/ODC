model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate= 0.3),
    Dense(128, activation= 'relu'),
    Dropout(rate= 0.25),
    Dense(num_classes, activation= 'softmax')
])

model.compile(Adamax(learning_rate= 0.001),
              loss= 'binary_crossentropy',
              metrics= ['accuracy'])
model.build((None, 224, 224, 3))
model.summary()

history = model.fit(tr_gen,
                 epochs=7,
                 validation_data=valid_gen,
                 shuffle= True)

test_loss, test_acc = model.evaluate(ts_gen, verbose=1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend(['тренировочная выборка', 'оценочная выборка'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь')
plt.ylabel('Потери')
plt.xlabel('Эпоха')
plt.legend(['тренировочная выборка', 'оценочная выборка'], loc='upper left')
plt.show()

print(test_acc)

pred = model.predict(ts_gen)
pred = np.argmax(pred, axis=1)

labels = (tr_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred2 = [labels[k] for k in pred]

from sklearn.metrics import accuracy_score

y_test = test_df['Class']
print(classification_report(y_test, pred2))
print("Accuracy of the Model:","{:.1f}%".format(accuracy_score(y_test, pred2)*100))

classes=list(tr_gen.class_indices.keys())
print (classes)


class_dict = ts_gen.class_indices
classes = list(class_dict.keys())
images, labels = next(ts_gen)
x = 6

plt.figure(figsize=(30, 30))

plt.subplot(4,4,i+1)
image = images[x] / 255
plt.imshow(image)
index = np.argmax(labels[i])
class_name = classes[index]
plt.title(class_name, fontsize=20)
plt.axis('off')


plt.figure(figsize=(30,30))
sample = tf.expand_dims(images[x], 0)
pred = model.predict(sample)

score = tf.nn.softmax(pred[0])
title = "{}".format(classes[np.argmax(score)])

ax = plt.subplot(4,4,i+1)
image = images[x] / 255
plt.imshow(image)
plt.title(title)
plt.axis('off')
