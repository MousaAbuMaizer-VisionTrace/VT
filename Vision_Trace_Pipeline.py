from flask import Flask, request, jsonify
import base64
from zipfile import ZipFile
import io
import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, concatenate
from tensorflow.keras import backend as K
import logging
from pathlib import Path
import json
import shutil
from aiofiles import open as async_open
from aiofiles.os import remove as async_remove
from flask import send_from_directory
import requests
import tempfile

logging.basicConfig(filename='application.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def load_images_from_folder(folder_path, image_size=(256,256), batch_size=32):
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
    batch_img_tensors = []
    for img_path in image_paths:
        if img_path.endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(img_path):
            img_tensor = load_and_preprocess_image(img_path, image_size)
            batch_img_tensors.append(img_tensor[0])

        if len(batch_img_tensors) == batch_size:
            yield np.array(batch_img_tensors)
            batch_img_tensors = []

    if batch_img_tensors:
        yield np.array(batch_img_tensors)

def load_and_preprocess_image(img_path, size=(256, 256), is_color=True):
    if is_color:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.merge([img, img, img])

    img = cv2.resize(img, size)
    img = np.expand_dims(img, axis=0) #/ 255.0
    return img

def save_results(output_folder, img_path, mask, max_voxel_img_path, max_voxel_mask):
    basename = os.path.basename(img_path)
    output_path = os.path.join(output_folder, basename)
    cv2.imwrite(output_path, mask)

    if max_voxel_img_path:
        print(f"Image with most tumor voxels: {max_voxel_img_path}, with {np.count_nonzero(max_voxel_mask)} voxels.")
        img_basename = os.path.basename(max_voxel_img_path)
        img_save_path = os.path.join(output_folder, 'highest_voxels', img_basename)
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        cv2.imwrite(img_save_path, cv2.imread(max_voxel_img_path, cv2.IMREAD_GRAYSCALE))
        mask_save_path = os.path.join(output_folder, 'highest_voxels', 'mask_' + img_basename)
        #cv2.imwrite(mask_save_path, max_voxel_mask)

def classify_images_batch(folder_path, model_path, image_size=(256,256), batch_size=32):
    try:
        model = load_model(model_path)
        tumor_image_paths = []
        print("Starting the Classification Process...")
        image_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
        for i, batch_img_tensors in enumerate(load_images_from_folder(folder_path, image_size, batch_size)):
            predictions = model.predict(batch_img_tensors)
            for j, pred in enumerate(predictions):
                if pred > 0.5:
                    tumor_image_paths.append(image_paths[i * batch_size + j])

        return tumor_image_paths
    except Exception as e:
        logging.error(f"Error loading classification model: {str(e)}")
        raise

def predict_images(input_folder, output_folder, model_path, tumor_image_paths):
    try:
        SIZE_Y = 256
        SIZE_X = 256
        img_size = (256, 256, 3)
        n_classes = 2

        unet_model = built_unet_model(img_size, n_classes)
        unet_model.load_weights(model_path)

        max_voxel_count = 0
        max_voxel_img_path = None
        max_voxel_mask = None

        if tumor_image_paths:
            print("Tumor Found, Starting the Segmentation Process...")

            for img_path in tumor_image_paths:
                img = load_and_preprocess_image(img_path)
                pred = unet_model.predict(img)

                mask = np.argmax(pred, axis=-1).astype(np.uint8)
                mask = np.squeeze(mask, axis=0) * 255

                voxel_count = np.count_nonzero(mask)
                if voxel_count > max_voxel_count:
                    max_voxel_count = voxel_count
                    max_voxel_img_path = img_path
                    max_voxel_mask = mask

            if max_voxel_img_path:
                print(f"Image with most tumor voxels: {max_voxel_img_path}, with {max_voxel_count} voxels.")
                img_basename = os.path.basename(max_voxel_img_path)
                img_save_path = os.path.join(output_folder, img_basename)
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                cv2.imwrite(img_save_path, cv2.imread(max_voxel_img_path, cv2.IMREAD_GRAYSCALE))
                mask_save_path = os.path.join(output_folder, 'mask_' + img_basename)
                #cv2.imwrite(mask_save_path, max_voxel_mask)
        else:
            print("No Tumor has been found.")
        
        return max_voxel_img_path

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise



####################  U_Net Model Architecture  ####################

def dice_coef(y_true, y_pred, smooth = 1.0):
    class_num = 2
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

def double_conv_block(x,n_filters):
    x=Conv2D(n_filters,3,padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(n_filters,3,padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    return x

def downsample_block(x, n_filters):
    f=double_conv_block(x, n_filters)
    p=MaxPool2D(2)(f)
    return f, p

def upsample_block(x, conv_features,n_filters):
    x=Conv2DTranspose(n_filters,3,2, padding="same")(x)
    x=concatenate([x, conv_features])
    x=double_conv_block(x, n_filters)
    return x

def built_unet_model(img_size, num_classes):
    input =Input(shape=img_size)
    f1,p1=downsample_block(input,64)
    f2,p2=downsample_block(p1,128)
    f3,p3=downsample_block(p2,256)
    f4,p4=downsample_block(p3,512)
    bottleneck=double_conv_block(p4,1024)
    u6=upsample_block(bottleneck,f4,512)
    u7=upsample_block(u6,f3,256)
    u8=upsample_block(u7,f2,128)
    u9=upsample_block(u8,f1,64)

    output=Conv2D(num_classes, 1, padding = "same", activation='softmax')(u9)
    unet_model=tf.keras.Model(input,output,name="U-Net")
    return unet_model

#########################################################################

############################# Flask Code ################################
app = Flask(__name__, static_folder='/var/www/html/VT/Processed-Scans')

@app.route('/app/process/<id>/<filename>', methods=['GET'])
def send_image(id, filename):
    print(f"Attempting to send image for ID: {id}, Filename: {filename}")  # This will log in the console
    try:
        return send_from_directory(os.path.join('/var/www/html/VT/Processed-Scans', id), filename)
    except Exception as e:
        print(f"Error sending image: {e}")
        return "Image not found", 404

@app.route('/app/process', methods=['POST'])
async def vision_trace():
    data = request.get_json(force=True)

    if 'id' not in data or 'zip_file_path' not in data:
        return jsonify({'error': 'Missing id or zip_file_path'}), 400

    id = data['id']
    zip_file_path = data['zip_file_path']
    root_folder = os.path.join('/var/www/html/VT/Scans', id)

    # Ensure the directory exists
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    # If it's a public link, download to a temp location and get original zip filename from URL
    original_zip_name = None
    if zip_file_path.startswith("http"):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        response = requests.get(zip_file_path, stream=True)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download file from provided URL'}), 400
        for chunk in response.iter_content(chunk_size=128):
            temp_file.write(chunk)
        temp_file.close()
        zip_file_path = temp_file.name
        original_zip_name = os.path.basename(response.url).split("?")[0]  # Get filename, discard query parameters

    # Extract and then remove the zip file if it's a temp file
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(root_folder)
    if zip_file_path == temp_file.name:
        os.remove(zip_file_path)

    # Use original_zip_name if it exists (i.e. was a URL download), otherwise derive from path
    zip_name = original_zip_name or Path(zip_file_path).stem
    potential_subfolder = os.path.join(root_folder, zip_name)
    images_folder = potential_subfolder if os.path.exists(potential_subfolder) else root_folder

    # Set the output_folder directly to the id
    output_folder = os.path.join('/var/www/html/VT/Processed-Scans', id)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    try:
        classification_model_path = '/var/www/html/VT/Model_Save_Weight/Classification_model.hdf5'
        segmentation_model_path = '/var/www/html/VT/Model_Save_Weight/Segmentation_model.hdf5'

        tumor_image_paths = classify_images_batch(images_folder, classification_model_path)
        max_voxel_img_path = predict_images(images_folder, output_folder, segmentation_model_path, tumor_image_paths)

        image_path = os.path.join(output_folder, os.path.basename(max_voxel_img_path)) if max_voxel_img_path else None

        response = {}

        if image_path is not None:
            relative_path = os.path.join('vision_trace', id, os.path.basename(max_voxel_img_path)).replace('\\', '/')
            response.update({
                'tumor_image_link': f"http://127.0.0.1:5000/{relative_path}"
            })
        else:
            response.update({
                'message': "No tumor image to send."
            })

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

#########################################################################
