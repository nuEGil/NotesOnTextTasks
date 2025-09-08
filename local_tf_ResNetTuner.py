import argparse
import pandas as pd 
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

''' the new way to do segmentation would be to take a newtork architecture, and pretrain it with contrastive learning right. 

This is just to demonstrate the ability to load pretrained models, and fine tune them. 
'''

# # To use a specific GPU (e.g., the first one, index 0)
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.set_visible_devices(gpus[0], 'GPU')
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

def get_args():
    parser = argparse.ArgumentParser(description="Extract image patches with stride.")
    parser.add_argument("--file", type=str, required=False, help="Path to the input file")
    parser.add_argument("--labeled_file", type=str, required=False, help="Path to the input file")
    return parser.parse_args()

# this section needs to run first so that you can get some preliminary labels before anything
def load_baseimg(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.expand_dims(img, axis =0)
    print('imshape ',img.shape)
    return preprocess_input(img)

def preliminary_labels():
    # base model is ResNet50
    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=True, input_shape=(224,224,3))
    # read the patch data
    patch_data = pd.read_csv(args.file)
    
    #without training - assign a label to every patch 
    labels = []
    for ii in range(patch_data.shape[0]):
        path_ = patch_data["patch_path"].iloc[ii]
        patch_ = load_baseimg(path_, 224)
        labels.append(np.argmax(base.predict(patch_)[0]))
    # print(labels)
    patch_data["labels"] = labels
    
    # get the unique classes 
    unique_classes = patch_data["labels"].unique()
    num_classes = len(unique_classes)

    # make a mapping {class_name: index}
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

    # update df with new integer labels
    patch_data["label_idx"] = patch_data["labels"].map(class_to_idx)
    # now save the csv for labeled images. 
    patch_data.to_csv(args.file.replace('.csv', '_labeled.csv'))

# ok then run this section to actually train a new model 
def df_to_dataset(df, image_size=64, batch_size=32, shuffle=True):
    paths, labels = df["patch_path"].values, df["label_idx"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        return preprocess_input(img), label
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(len(df))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model(file_w_labels):
    # split the data sets
    patch_data = pd.read_csv(file_w_labels)
    unique_classes = patch_data["labels"].unique()
    num_classes = len(unique_classes)

    # make a mapping {class_name: index}
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

    # update df with new integer labels
    patch_data["label_idx"] = patch_data["labels"].map(class_to_idx)


    train_df = patch_data.sample(frac=0.9, random_state=42)
    test_df = patch_data.drop(train_df.index)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}") 

    # load the model    
    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(64,64,3))
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # train the model
    train_ds = df_to_dataset(train_df)
    test_ds  = df_to_dataset(test_df, shuffle=False)
    history = model.fit(train_ds, validation_data=test_ds, epochs=5)

if __name__ == "__main__":
    args = get_args()
    # Load ResNet50 pretrained on ImageNet
    ## get the labels if you need to
    preliminary_labels()
    
    # # train 
    # train_model(args.labeled_file)
    

