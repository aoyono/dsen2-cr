import csv
import os
import os.path

import matplotlib
import numpy as np
import rasterio
import scipy.signal as scisig
from matplotlib import pyplot as plt
from tensorflow.compat.v1 import keras

from sen2cr.tools.feature_detectors import get_cloud_cloudshadow_mask
from pathlib import Path


def get_scene_data(scene_name, base_dir='data', cloudy_dir='cloudy_s2', cloudless_dir='cloudless_s2', s1_dir='s1'):
    """Get the data for a given scene"""
    base_dir = Path(base_dir)
    data = {}
    for data_type in (cloudy_dir, cloudless_dir, s1_dir):
        path = base_dir / data_type / scene_name
        if path.exists():
            with rasterio.open(path, 'r') as ds:
                img = ds.read()
                # Replace NaNs with the mean of the image
                img[np.isnan(img)] = np.nanmean(img)
                data[data_type] = img
    return data


def preprocess_scene_data(scene_data, sar_clip_min=None, sar_clip_max=None, sar_max_value=2., s2_clip_min=None, s2_clip_max=None, s2_scale=2000., channel_last=False):
    preproc_scene_data = {}
    for data_type, img in scene_data.items():
        if data_type == 's1':
            preproc_scene_data[data_type] = preprocess_s1_data(img.astype('float32'), sar_clip_min, sar_clip_max, sar_max_value, channel_last)
        elif data_type in ('cloudy_s2', 'cloudless_s2'):
            preproc_scene_data[data_type] = preprocess_s2_data(img.astype('float32'), s2_clip_min, s2_clip_max, s2_scale, channel_last)
    return preproc_scene_data


def get_scene_data_batch(scene_names, base_dir='data', cloudy_dir='cloudy_s2', cloudless_dir='cloudless_s2', s1_dir='s1', sar_clip_min=None, sar_clip_max=None, sar_max_value=2., s2_clip_min=None, s2_clip_max=None, s2_scale=2000.,):
    scene_img_data = [get_scene_data(scene_name, base_dir, cloudy_dir, cloudless_dir, s1_dir) for scene_name in scene_names]
    scene_data = [
        preprocess_scene_data(
            img_data, sar_clip_min, sar_clip_max, sar_max_value, s2_clip_min, s2_clip_max, s2_scale,
        )
        for img_data in scene_img_data
    ]
    batch_length = len(scene_names)
    s2_data_dims = scene_data[0]['cloudy_s2'].shape
    s1_data_dims = scene_data[0]['s1'].shape
    batch = {
        cloudy_dir: np.empty_like((batch_length, *s2_data_dims)).astype('float32'),
        cloudless_dir: np.empty_like((batch_length, *s2_data_dims)).astype('float32'),
        s1_dir: np.empty_like((batch_length, *s1_data_dims)).astype('float32'),
    }
    for i, data in enumerate(scene_data):
        for data_type, img in data.items():
            batch[data_type][i] = img
    return batch


def save_model_output_batch(model_output, scene_names, output_dir='output', input_base_dir='data', postprocess=True, s2_scale=2000., channel_last=False):
    for i, scene_name in enumerate(scene_names):
        if postprocess:
            data = postprocess_model_output(model_output[i], s2_scale=s2_scale, channel_last=channel_last)
        save_model_output(data, scene_name, output_dir, input_base_dir)


def extract_rgb(data, channel_last=False, rgb_bands=(3, 2, 1)):
    original_dtype = data.dtype
    if original_dtype in ('uint8', 'uint16'):
        data = data.astype('float32')
    if channel_last:
        data = np.transpose(data, (2, 0, 1))[rgb_bands, :, :]
    else:
        data = data[:, :, rgb_bands]
    data -= np.nanmin(data)
    if np.nanmax(data) == 0:
        data = 255. * np.ones_like(data)
    else:
        data *= 1. / np.nanmax(data)
    data[np.isnan(data)] = np.nanmean(data)
    return data.astype(original_dtype)



def postprocess_model_output(model_output, s2_scale=2000., channel_last=False, tci=False, rgb_bands=(3, 2, 1)):
    model_output *= s2_scale
    if tci:
        return extract_rgb(model_output, channel_last=channel_last, rgb_bands=rgb_bands)
    if channel_last:
        model_output = np.transpose(model_output, (2, 0, 1))
    return model_output


def save_model_output(model_output, scene_name, output_dir='output', input_base_dir='data', post_process=False, s2_scale=2000., channel_last=False, tci=False, rgb_bands=(3, 2, 1)):
    output_dir = Path(output_dir)
    input_dir = Path(input_base_dir) / 'cloudy_s2'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / scene_name
    input_profile = rasterio.open(input_dir / scene_name).meta
    input_profile.update(dtype='float32')
    if post_process:
        model_output = postprocess_model_output(model_output, s2_scale=s2_scale, channel_last=channel_last, tci=tci, rgb_bands=rgb_bands)
        if tci:
            input_profile.update(count=3, dtype='uint8')
    with rasterio.open(output_path, 'w', **input_profile) as ds:
        ds.write(model_output)


def preprocess_s1_data(s1_data, clip_min=None, clip_max=None, sar_max_value=2., channel_last=False):
    if clip_min is None:
        clip_min = np.array([[[-25.0]], [[-32.5]]])
    if clip_max is None:
        clip_max = np.array([[[0.]]] * 2)
    data = np.clip(s1_data, clip_min, clip_max)
    data -= clip_min
    data *= sar_max_value / (clip_max - clip_min)
    if s1_data.shape[0] == 2 and channel_last:
        data = np.transpose(data, (1, 2, 0))
    return data


def preprocess_s2_data(s2_data, clip_min=None, clip_max=None, s2_scale=2000., channel_last=False):
    if clip_min is None:
        clip_min = np.array([[[0.]]] * 13)
    if clip_max is None:
        clip_max = np.array([[[10000.]]] * 13)
    data = np.clip(s2_data, clip_min, clip_max)
    data /= s2_scale
    if s2_data.shape[0] == 13 and channel_last:
        data = np.transpose(data, (1, 2, 0))
    return data


def make_dir(dir_path):
    if os.path.isdir(dir_path):
        print("WARNING: Folder {} exists and content may be overwritten!")
    else:
        os.makedirs(dir_path)

    return dir_path


def get_train_val_test_filelists(listpath):
    with open(listpath) as f:
        filelist = csv.reader(f, delimiter='\t')

        train_filelist = []
        val_filelist = []
        test_filelist = []
        for line_entries in filelist:
            if line_entries[0] == '1':
                train_filelist.append(line_entries)
            if line_entries[0] == '2':
                val_filelist.append(line_entries)
            if line_entries[0] == '3':
                test_filelist.append(line_entries)

        return train_filelist, val_filelist, test_filelist


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_info_quartet(ID, predicted_images_path, input_data_folder):
    scene_name = ID[4]
    filepath_sar = os.path.join(input_data_folder, ID[1], ID[4]).lstrip()
    filepath_cloudFree = os.path.join(input_data_folder, ID[2], ID[4]).lstrip()
    filepath_cloudy = os.path.join(input_data_folder, ID[3], ID[4]).lstrip()

    return scene_name[:-4], filepath_sar, filepath_cloudFree, filepath_cloudy


def get_rgb_preview(r, g, b, sar_composite=False):
    if not sar_composite:

        # stack and move to zero
        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)

        # treat saturated images, scale values
        if np.nanmax(rgb) == 0:
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))

        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)

        return rgb.astype(np.uint8)

    else:
        # generate SAR composite
        HH = r
        HV = g

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6

        rgb = np.dstack((np.zeros_like(HH), HH, HV))

        return rgb.astype(np.uint8)


def get_raw_data(path):
    with rasterio.open(path, driver='GTiff') as src:
        image = src.read()

    # checkimage for nans
    image[np.isnan(image)] = np.nanmean(image)

    return image.astype('float32')


def get_preview(file, predicted_file, bands, brighten_limit=None, sar_composite=False):
    if not predicted_file:
        with rasterio.open(file) as src:
            r, g, b = src.read(bands)
    else:
        # file is actually the predicted array
        r = file[bands[0] - 1]
        g = file[bands[1] - 1]
        b = file[bands[2] - 1]

    if brighten_limit is None:
        return get_rgb_preview(r, g, b, sar_composite)
    else:
        r = np.clip(r, 0, brighten_limit)
        g = np.clip(g, 0, brighten_limit)
        b = np.clip(b, 0, brighten_limit)
        return get_rgb_preview(r, g, b, sar_composite)


def generate_output_images(predicted, ID, predicted_images_path, input_data_folder, cloud_threshold):
    scene_name, filepath_sar, filepath_cloudFree, filepath_cloudy = get_info_quartet(ID,
                                                                                     predicted_images_path,
                                                                                     input_data_folder)

    print("Generating quartet for ", scene_name)

    sar_preview = get_preview(filepath_sar, False, [1, 2, 2], sar_composite=True)

    opt_bands = [4, 3, 2]  # R, G, B bands (S2 channel numbers)
    cloudFree_preview = get_preview(filepath_cloudFree, False, opt_bands, brighten_limit=2000)
    cloudy_preview = get_preview(filepath_cloudy, False, opt_bands)
    cloudy_preview_brightened = get_preview(filepath_cloudy, False, opt_bands, brighten_limit=2000)

    predicted_preview = get_preview(predicted, True, opt_bands, 2000)

    cloud_mask = get_cloud_cloudshadow_mask(get_raw_data(filepath_cloudy), cloud_threshold)

    save_single_images(sar_preview, cloudy_preview, cloudFree_preview, predicted_preview, cloudy_preview_brightened,
                       cloud_mask, predicted_images_path, scene_name)

    return


def save_single_image(image, out_path, name):
    plt.figure(frameon=False)
    plt.imshow(image)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(os.path.join(out_path, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def save_single_cloudmap(image, out_path, name):
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'white'])

    bounds = [-1, -0.5, 0.5, 1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure()
    plt.imshow(image, cmap=cmap, norm=norm, vmin=-1, vmax=1)

    cb = plt.colorbar(aspect=40, pad=0.01)
    cb.ax.yaxis.set_tick_params(pad=0.9, length=2)

    cb.ax.yaxis.set_ticks([0.33 / 2, 0.5, 1 - 0.33 / 2])
    cb.ax.yaxis.set_ticklabels(['Shadow', 'Clear', 'Cloud'])

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(os.path.join(out_path, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def save_single_images(sar_preview, cloudy_preview, cloudFree_preview, predicted_preview, cloudy_preview_brightened,
                       cloud_mask, predicted_images_path, scene_name):
    out_path = make_dir(os.path.join(predicted_images_path, scene_name))

    save_single_image(sar_preview, out_path, "inputsar")
    save_single_image(cloudy_preview, out_path, "input")
    save_single_image(cloudFree_preview, out_path, "inputtarg")
    save_single_image(predicted_preview, out_path, "inputpred")
    save_single_image(cloudy_preview_brightened, out_path, "inputbr")
    save_single_cloudmap(cloud_mask, out_path, "cloudmask")

    return


def process_predicted(predicted, ID, predicted_images_path, scale, cloud_threshold, input_data_folder):
    for i, data_image in enumerate(predicted):
        data_image *= scale
        generate_output_images(data_image, ID[i], predicted_images_path, input_data_folder, cloud_threshold)

    return


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class DataGenerator(keras.utils.Sequence):
    """DataGenerator for Keras routines."""

    def __init__(self,
                 list_IDs,
                 batch_size=32,
                 input_dim=((13, 256, 256), (2, 256, 256)),
                 scale=2000,
                 shuffle=True,
                 include_target=True,
                 data_augmentation=False,
                 random_crop=False,
                 crop_size=128,
                 clip_min=None,
                 clip_max=None,
                 input_data_folder='./',
                 use_cloud_mask=True,
                 max_val_sar=5,
                 cloud_threshold=0.2
                 ):

        if clip_min is None:
            clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.nr_images = len(self.list_IDs)
        self.indexes = np.arange(self.nr_images)
        self.scale = scale
        self.shuffle = shuffle
        self.include_target = include_target
        self.data_augmentation = data_augmentation
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.max_val = max_val_sar

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.input_data_folder = input_data_folder
        self.use_cloud_mask = use_cloud_mask
        self.cloud_threshold = cloud_threshold

        self.augment_rotation_param = np.repeat(0, self.nr_images)
        self.augment_flip_param = np.repeat(0, self.nr_images)
        self.random_crop_paramx = np.repeat(0, self.nr_images)
        self.random_crop_paramy = np.repeat(0, self.nr_images)

        self.on_epoch_end()

        print("Generator initialized")

    def __len__(self):
        """Gets the number of batches per epoch"""
        return int(np.floor(self.nr_images / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch from shuffled indices list
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if self.include_target:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                          self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                          self.random_crop_paramy[indexes])
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                       self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                       self.random_crop_paramy[indexes])
            return X

    def on_epoch_end(self):
        """Update indexes after each epoch."""

        if self.shuffle:
            np.random.shuffle(self.indexes)

        if self.data_augmentation:
            self.augment_rotation_param = np.random.randint(0, 4, self.nr_images)
            self.augment_flip_param = np.random.randint(0, 3, self.nr_images)

        if self.random_crop:
            self.random_crop_paramx = np.random.randint(0, self.crop_size, self.nr_images)
            self.random_crop_paramy = np.random.randint(0, self.crop_size, self.nr_images)
        return

    def __data_generation(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                          random_crop_paramx_temp, random_crop_paramy_temp):

        input_opt_batch, cloud_mask_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp,
                                                           augment_flip_param_temp, random_crop_paramx_temp,
                                                           random_crop_paramy_temp, data_type=3)

        input_sar_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                         random_crop_paramx_temp, random_crop_paramy_temp, data_type=1)

        if self.include_target:
            output_opt_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                              random_crop_paramx_temp, random_crop_paramy_temp, data_type=2)
            if self.use_cloud_mask > 0:
                output_opt_cloud_batch = [np.append(output_opt_batch[sample], cloud_mask_batch[sample], axis=0) for
                                          sample in range(len(output_opt_batch))]
                output_opt_cloud_batch = np.asarray(output_opt_cloud_batch)
                return ([input_opt_batch, input_sar_batch], [output_opt_cloud_batch])
            else:
                return ([input_opt_batch, input_sar_batch], [output_opt_batch])
        elif not self.include_target:
            # for prediction step where target is predicted
            return ([input_opt_batch, input_sar_batch])

    def get_image_data(self, paramx, paramy, path):
        # with block not working with window kw
        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read(window=((paramx, paramx + self.crop_size), (paramy, paramy + self.crop_size)))
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts
        return image

    def get_opt_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        return image.astype('float32')

    def get_sar_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        medianfilter_onsar = False
        if medianfilter_onsar:
            image[0] = scisig.medfilt2d(image[0], 7)
            image[1] = scisig.medfilt2d(image[1], 7)

        return image.astype('float32')

    def get_data_image(self, ID, data_type, paramx, paramy):

        data_path = os.path.join(self.input_data_folder, ID[data_type], ID[4]).lstrip()

        if data_type == 2 or data_type == 3:
            data_image = self.get_opt_image(data_path, paramx, paramy)
        elif data_type == 1:
            data_image = self.get_sar_image(data_path, paramx, paramy)
        else:
            print('Error! Data type invalid')

        return data_image

    def get_normalized_data(self, data_image, data_type):

        shift_data = False

        shift_values = [[0, 0], [1300., 981., 810., 380., 990., 2270., 2070., 2140., 2200., 650., 15., 1600., 680.],
                        [1545., 1212., 1012., 713., 1212., 2476., 2842., 2775., 3174., 546., 24., 1776., 813.]]

        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
            if shift_data:
                data_image -= self.max_val / 2
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                if shift_data:
                    data_image[channel] -= shift_values[data_type - 1][channel]

            data_image /= self.scale

        return data_image

    def get_batch(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp, random_crop_paramx_temp,
                  random_crop_paramy_temp, data_type):

        if data_type == 1:
            dim = self.input_dim[1]
        else:
            dim = self.input_dim[0]

        batch = np.empty((self.batch_size, *dim)).astype('float32')
        cloud_mask_batch = np.empty((self.batch_size, self.input_dim[0][1], self.input_dim[0][2])).astype('float32')

        for i, ID in enumerate(list_IDs_temp):

            data_image = self.get_data_image(ID, data_type, random_crop_paramx_temp[i], random_crop_paramy_temp[i])
            if self.data_augmentation:
                if not augment_flip_param_temp[i] == 0:
                    data_image = np.flip(data_image, augment_flip_param_temp[i])
                if not augment_rotation_param_temp[i] == 0:
                    data_image = np.rot90(data_image, augment_rotation_param_temp[i], (1, 2))

            if data_type == 3 and self.use_cloud_mask:
                cloud_mask = get_cloud_cloudshadow_mask(data_image, self.cloud_threshold)
                cloud_mask[cloud_mask != 0] = 1
                cloud_mask_batch[i,] = cloud_mask

            data_image = self.get_normalized_data(data_image, data_type)

            batch[i,] = data_image

        cloud_mask_batch = cloud_mask_batch[:, np.newaxis, :, :]

        if data_type == 3:
            return batch, cloud_mask_batch
        else:
            return batch
