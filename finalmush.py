
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import tensorflow as tf
from glob import glob

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model


from tensorflow.keras.applications.resnet_v2 import ResNet152V2,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential


from pyswarms.single.global_best import GlobalBestPSO
from tensorflow.keras.callbacks import EarlyStopping


train_path = 'PlantVillage1/train'
test_path = 'PlantVillage1/val'

image_size = [224,224]

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   shear_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


train_set = train_datagen.flow_from_directory('PlantVillage1/train',
                                              target_size=(224, 224),
                                              batch_size = 32,
                                              class_mode = 'categorical')
                                          
test_set = test_datagen.flow_from_directory('PlantVillage1/val',
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


resnet152V2  = ResNet152V2(input_shape = image_size + [3], weights = 'imagenet', include_top = False)


for layer in resnet152V2.layers:
    layer.trainable = False


folders = glob('PlantVillage1/train/*')

resnet152V2.output


x = Flatten()(resnet152V2.output)


len(folders)


prediction = Dense(len(folders), activation='softmax')(x)


model = Model(inputs = resnet152V2.input, outputs = prediction)

model.summary()

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),loss='categorical_crossentropy', metrics=["accuracy"])


def fitness_function(params):
    learning_rate = params[0]
    batch_size = int(params[1])
    epochs = int(params[2])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    r = model.fit_generator(train_set, 
                            validation_data=test_set, 
                            epochs=epochs, 
                            steps_per_epoch=len(train_set), 
                            validation_steps=len(test_set),
                            batch_size=batch_size,
                            verbose=0)
    accuracy = r.history['accuracy']
    return  accuracy


bounds = [(0.0001, 0.01),  
          (16, 64),       
          (5, 20)]
types = [float, int, int]
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = GlobalBestPSO(n_particles=10, dimensions=3, bounds=bounds, options=options)
optimizer.optimize(fitness_function, iters=10)


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()
plt.savefig('lossVal_loss_ResNet152v2')


plt.plot(r.history['accuracy'], label= 'accuracy')
plt.plot(r.history['val_accuracy'], label='validation_accuracy')
plt.legend()
plt.show()
plt.savefig('acurracyVal_accuracy_ResNet152v2')



y_pred = model.predict(test_set)


len(y_pred)


y_pred.shape


final_pred = np.argmax(y_pred, axis=1)

len(final_pred)

final_pred.shape


model.save('model_ResNet152v2new.h5')

model=load_model('model_ResNet152v2new.h5')


from PIL import Image
img = Image.open('PlantVillage1/val/Potato___healthy/1ae826e2-5148-47bd-a44c-711ec9cc9c75___RS_HL 1954.JPG')
img = img.resize((224, 224)) 


from tensorflow.keras.preprocessing.image import img_to_array

x = img_to_array(img)

x.shape

x=x/255

x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


model.predict(img_data)

a=np.argmax(model.predict(img_data), axis=1)

a==1


classes=list(train_set.class_indices.keys())
print(classes)

def prepare(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    x = np.asarray(img)
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    return x


img_url='/Users/book/tensorflow-test/project1/PlantVillage/Tomato_Bacterial_spot/0b37769a-a451-4507-a236-f46348e3a9ac___GCREC_Bact.Sp 3265.JPG'
result_resnet152v2 = model.predict([prepare(img_url)])
disease=Image.open(img_url)


plt.imshow(disease)
classresult=np.argmax(result_resnet152v2,axis=1)
print(classes[classresult[0]])






