
# You find the model desired in the folder named Models
Model_name = 'Model_Name'
loaded_model = tf.keras.models.load_model('../Models/f'{Model_name}')


fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15, 6))

# Plotting the training and validation loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_ylabel('Loss')
ax1.legend()

# Plotting the training and validation accuracy
ax2.plot(history.history['dice_coef'], label='Training Accuracy')
ax2.plot(history.history['val_dice_coef'], label='Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Dice_coef')
ax2.legend()

plt.suptitle('Training and Validation Metrics')
plt.show()