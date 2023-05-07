
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import tensorflow as tf
from glob import glob

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model



train_path = 'PlantVillage1/train'
test_path = 'PlantVillage1/val'

image_size = [224,224]

inception = InceptionV3(input_shape=image_size + [3], weights='imagenet', include_top=False)


for layer in inception.layers:
    layer.trainable = False


folders = glob('PlantVillage1/train/*')


inception.output

x = Flatten()(inception.output)

len(folders)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs = inception.input, outputs = prediction)

model.summary()


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),loss='categorical_crossentropy', metrics=["accuracy"])


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



r = model.fit_generator(train_set, 
                        validation_data= test_set, 
                        epochs=50, 
                        steps_per_epoch= len(train_set), 
                        validation_steps = len(test_set))




plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()
plt.savefig('lossVal_loss_ResNet50')


plt.plot(r.history['accuracy'], label= 'accuracy')
plt.plot(r.history['val_accuracy'], label='validation_accuracy')
plt.legend()
plt.show()
plt.savefig('acurracyVal_accuracy_ResNet50')



model.save('model_inceptionv3.h5')

model=load_model('model_inceptionv3.h5')



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





