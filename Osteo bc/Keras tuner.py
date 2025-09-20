pip install -U keras-tuner

import keras_tuner
from keras_tuner.tuners import Hyperband
from tensorflow.keras import utils

def build_model(hp):
  model = Sequential([
      base_model,
      Flatten(),
      Dropout(0.3),
      Dense(units=hp.Int('units_hidden',
                                min_value=128,
                                max_value=600,
                                step=32),
                   activation=hp.Choice('activation',
                                        values=['selu',
                                                'gelu',
                                                'relu',
                                                'elu',
                                                'swish',
                                                'tanh'])),
                      Dropout(0.25),
      Dense(num_classes, activation='softmax')])

  model.compile(
      optimizer=hp.Choice('optimizer',
                          values=['nadam',
                                  'adadelta',
                                  'adagrad',
                                  'adam',
                                  'adamax',
                                  'ftrl',
                                  'rmsprop']),
      loss='categorical_crossentropy',
      metrics = ['accuracy'])
  return model

tuner = Hyperband(
    build_model,
    objective='val_accuracy',

    max_epochs=100,
    directory='test_directory_HB'
    )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search_space_summary()

tuner.search(
    tr_gen,
    validation_data=valid_gen,
    batch_size=batch_size,
    epochs = 30,
    callbacks=[stop_early]
)

tuner.results_summary()

models = tuner.get_best_models(num_models=1)

for model in models:
  model.summary(),
  model.evaluate(ts_gen)
  print()
