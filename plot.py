from keras.utils.vis_utils import plot_model

def plot_training_score(history):
  print('Availible variables to plot: {}'.format(history.history.keys()))
  # TODO: Visulize the plot, to be applied after training is complete

  #plt.plot(history.history['val_acc'], '--',history.history['acc']) DELER DENNE PRINTEN I 2 UNDER
  try:
    plt.plot(history.history['val_acc'], '--', label = 'Validation data')
  except:
    pass
  plt.plot(history.history['acc'], label = 'Testdata')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

  #plt.plot(history.history['val_loss'], '--',history.history['loss']) DELER DENNE PRINTEN I 2 UNDER
  try:
    plt.plot(history.history['val_loss'], '--', label = 'Validation data')
  except:
    pass
  plt.plot(history.history['loss'], label = 'Testdata')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
