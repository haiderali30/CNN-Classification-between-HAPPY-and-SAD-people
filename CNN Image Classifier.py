#!/usr/bin/env python
# coding: utf-8

# In[108]:


# !pip install tensorflow tensorflow-gpu opencv-python matplotlib


# In[4]:


# pip list


# In[109]:


# pip install tensorflow-gpu


# In[5]:


import tensorflow as tf
import os
import matplotlib.pyplot as plt


# In[6]:


gpus = tf.config.experimental.list_physical_devices('CPU')


# In[7]:


len(gpus)


# In[8]:


import cv2
import imghdr


# In[9]:


data_dir = "images"


# In[10]:


os.listdir(os.path.join(data_dir,'happpy'))


# In[11]:


image_extensions = ['jpeg','jpg','bmp','png']


# In[12]:


image_extensions[2]


# In[13]:


import os

happy_dir = "images/happpy"

# Get the list of files in the happy folder
happy_files = os.listdir(happy_dir)

# Iterate over each file in the happy folder
for file_name in happy_files:
    # Construct the full file path
    file_path = os.path.join(happy_dir, file_name)
    
    # Check if the file exists and is a file (not a directory)
    if os.path.isfile(file_path):
        # Get the size of the file in bytes
        file_size = os.path.getsize(file_path)
        
        # Check if the file size is less than 10 KB (10 * 1024 bytes)
        if file_size < 10 * 1024:
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {file_name}")


# In[14]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        print(image)


# In[15]:


img = cv2.imread(os.path.join('images','happpy','05-12-21-happy-people.jpg'))


# In[16]:


type(img)


# In[17]:


img.shape


# In[18]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[19]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_extensions:
                print("image not in extension list {}". format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("issue with image {}". format(image_path))
            


# In[ ]:





# # Loading DATA

# In[20]:


# tf.data.Dataset??


# In[21]:


# tf.keras.utils.image_dataset_from_directory??


# In[22]:


images = tf.keras.utils.image_dataset_from_directory('images')


# In[23]:


images


# In[24]:


iamges_iterator = images.as_numpy_iterator()


# In[25]:


iamges_iterator


# In[26]:


# get another batch 
batch = iamges_iterator.next()


# In[27]:


batch[0].shape


# In[28]:


batch[1]


# In[29]:


batch[0].min()


# In[30]:


batch[0].max()


# In[31]:


fig, ax = plt.subplots(ncols = 4, figsize = (20,20))

for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[32]:


scaled = batch[0] / 255


# In[33]:


# scaled


# In[34]:


scaled.min()


# In[35]:


scaled.max()


# # Preprocess DATA

# In[36]:


images = images.map(lambda x,y: (x/255,y))


# In[37]:


images.as_numpy_iterator().next()[0].min()


# In[38]:


scaled_iterator = images.as_numpy_iterator()


# In[39]:


batch = scaled_iterator.next()


# In[40]:


batch[0].max()


# In[41]:


fig, ax = plt.subplots(ncols = 4, figsize = (20,20))

for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# In[ ]:





# # Splitting the DATA

# In[42]:


len(images)


# In[43]:


train_size = int(len(images)*.7)
val_size = int(len(images)*.2)
test_size = int(len(images)*.1)


# In[44]:


val_size


# In[45]:


# Assuming len(images) gives the total number of images in your dataset
total_size = len(images)

# Define the proportions for training, validation, and test sets
train_proportion = 0.7  # 70% for training
val_proportion = 0.2    # 20% for validation
test_proportion = 0.1   # 10% for testing

# Calculate the sizes for training, validation, and test sets
train_size = int(total_size * train_proportion)
val_size = int(total_size * val_proportion)
test_size = total_size - train_size - val_size

print("Train size:", train_size)
print("Validation size:", val_size)
print("Test size:", test_size)


# In[46]:


train = images.take(train_size)
val = images.skip(train_size).take(val_size)
test = images.skip(train_size + val_size).take(test_size)


# In[47]:


len(val)


# In[48]:


import tensorflow as tf


# In[49]:


# Building DEEP LEARNING Model


# In[50]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[51]:


model = Sequential()


# In[52]:


model.add(Conv2D(16, (3,3), 1 , activation = 'relu', input_shape = (256,256,3)))
model.add (MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add (MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation = 'relu'))
model.add (MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add (Dense(1, activation = 'sigmoid'))


# In[53]:


model.compile(optimizer = 'adam',loss = tf.losses.BinaryCrossentropy(),metrics = ['accuracy'])
# model.compile(optimizer='adam', 
#               loss=tf.losses.BinaryCrossentropy(), 
#               metrics=['accuracy'])


# In[54]:


model.summary()


# In[55]:


13*13*16


# In[ ]:





# # Training

# In[56]:


logdir = 'logs'


# In[57]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[58]:


hist = model.fit(train, epochs=20, validation_data = val , callbacks =[tensorboard_callback])


# In[59]:


hist


# In[61]:


# hist.history


# In[ ]:





# # Plotting Performance

# In[62]:


fig = plt.figure()
plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('loss', fontsize = 20)
plt.legend(loc= 'upper left')
plt.show()


# In[63]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'teal', label = 'accuracy')
plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'val_accuracy')
fig.suptitle('accuracy', fontsize = 20)
plt.legend(loc= 'upper left')
plt.show()


# In[ ]:





# In[64]:


# Evaliating Performance


# In[65]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[66]:


pre = Precision()
rec = Recall()
acc = BinaryAccuracy()


# In[67]:


for batch in test.as_numpy_iterator():
    x,y = batch
    yhat = model.predict(x)
    pre.update_state(y,yhat)
    rec.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[68]:


len(test)


# In[69]:


import re
print(f'precision {pre.result().numpy() }, Recall: {rec.result().numpy() } , Accuracy: {acc.result().numpy() }')


# print(f'Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')


# In[70]:


import cv2
import matplotlib.pyplot as plt


# In[95]:


img = cv2.imread("happytest4.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[96]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[97]:


#predict
import numpy as np


# In[98]:


resize.shape


# In[99]:


np.expand_dims(resize,0).shape


# In[100]:


yhat  = model.predict(np.expand_dims(resize/255,0))


# In[101]:


yhat


# In[102]:


if yhat > 0.5:
    print ("person is SAD")

else:
    print("Person is HAPPY")
    


# In[ ]:





# # saving the model

# In[103]:


from tensorflow.keras.models import load_model



# In[104]:


model.save(os.path.join('models', 'happysadprecdictionmodel.h5'))


# In[105]:


new_model = load_model(os.path.join('models', 'happysadprecdictionmodel.h5'))


# In[106]:


yhatnew = new_model.predict(np.expand_dims(resize/255, 0))


# In[107]:


if yhatnew > 0.5:
    print ("person is SAD")

else:
    print("Person is HAPPY")
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




