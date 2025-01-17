import random
import os
import nibabel as nib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from PIL import Image


def get_filenames_in_directory(directory_path):
    return [
        file
        for file in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, file))
    ]


def process_images(
    volumes,
    segmentations,
    nii_volumes_path,
    nii_segmentation_path,
    output_volumes_path,
    output_segmentations_path,
    skip,
    images_num,
):

    images_num = min(images_num, len(volumes))

    args = [
        (
            volumes[i],
            segmentations[i],
            nii_volumes_path,
            nii_segmentation_path,
            output_volumes_path,
            output_segmentations_path,
            skip,
        )
        for i in range(images_num)
    ]

    if not os.path.exists(output_volumes_path):
        os.makedirs(output_volumes_path)
    if not os.path.exists(output_segmentations_path):
        os.makedirs(output_segmentations_path)

    with Pool(processes=min(os.cpu_count(), images_num)) as pool:
        pool.starmap(process_image_file, args)


def split_train_test(volumes_list, segmentation_list, test_count):
    test_indexes = random.sample(range(len(volumes_list)), test_count)

    test_volumes_list = [volumes_list[i] for i in test_indexes]
    test_segmentation_list = [segmentation_list[i] for i in test_indexes]

    train_volumes_list = [
        file for i, file in enumerate(volumes_list) if i not in test_indexes
    ]
    train_segmentation_list = [
        file for i, file in enumerate(segmentation_list) if i not in test_indexes
    ]

    return (
        train_volumes_list,
        train_segmentation_list,
        test_volumes_list,
        test_segmentation_list,
    )


def crop_image(output_file_path):
    with Image.open(output_file_path) as img:
        if img.size != (389, 389):
            cropped_img = img.crop((0, 0, 389, 389))
            cropped_img.save(output_file_path)


def save_images(
    image_data,
    segmentation_data,
    skip,
    png_volumes_path,
    png_segmentation_path,
    volume_file,
    segmentation_file,
    threshold=100,
):
    for index in range(0, segmentation_data.shape[2], skip):
        data_sum = np.sum(segmentation_data[:, :, index])
        if data_sum < threshold:
            continue

        plt.imshow(segmentation_data[:, :, index], cmap="gray")
        plt.axis("off")
        output_file_path = os.path.join(
            png_segmentation_path, f"{segmentation_file}_{index}.png"
        )
        plt.savefig(output_file_path, bbox_inches="tight", facecolor="black")
        plt.close()
        crop_image(output_file_path)

        plt.imshow(image_data[:, :, index], cmap="gray")
        plt.axis("off")
        output_file_path = os.path.join(png_volumes_path, f"{volume_file}_{index}.png")
        plt.savefig(output_file_path, bbox_inches="tight", facecolor="black")
        plt.close()
        crop_image(output_file_path)


def process_image_file(
    volume_file,
    segmentation_file,
    nii_volumes_path,
    nii_segmentation_path,
    png_volumes_path,
    png_segmentation_path,
    skip,
):
    if volume_file != segmentation_file:
        return

    image_file_path = os.path.join(nii_volumes_path, volume_file)
    image_data_nii = nib.load(image_file_path)
    image_data = image_data_nii.get_fdata()

    segmentation_file_path = os.path.join(nii_segmentation_path, segmentation_file)
    segmentation_nii = nib.load(segmentation_file_path)
    segmentation_data = segmentation_nii.get_fdata()

    try:
        assert (
            image_data.shape == segmentation_data.shape
        ), "The shapes of image and segmentation data are not the same"
    except AssertionError as e:
        print(f"Warning: {e}")
        return

    save_images(
        image_data,
        segmentation_data,
        skip,
        png_volumes_path,
        png_segmentation_path,
        volume_file,
        segmentation_file,
    )


def create_sets(
    create_images,
    skip,
    test_nii_files,
    nii_directory,
    png_directory,
    train_dir,
    test_dir,
    volumes_dir,
    segmentation_dir,
):

    nii_volumes_path = os.path.join(nii_directory, volumes_dir)
    nii_segmentation_path = os.path.join(nii_directory, segmentation_dir)
    png_train_volumes_path = os.path.join(png_directory, train_dir, volumes_dir)
    png_train_segmentation_path = os.path.join(
        png_directory, train_dir, segmentation_dir
    )
    png_test_volumes_path = os.path.join(png_directory, test_dir, volumes_dir)
    png_test_segmentation_path = os.path.join(png_directory, test_dir, segmentation_dir)

    volumes_list = get_filenames_in_directory(nii_volumes_path)
    segmentation_list = get_filenames_in_directory(nii_segmentation_path)

    volumes_list, segmentation_list, test_volumes_list, test_segmentation_list = (
        split_train_test(volumes_list, segmentation_list, test_nii_files)
    )

    if create_images != None:
        images_num = create_images
    else:
        images_num = len(volumes_list)

    if skip == None:
        skip = 1

    process_images(
        volumes_list,
        segmentation_list,
        nii_volumes_path,
        nii_segmentation_path,
        png_train_volumes_path,
        png_train_segmentation_path,
        skip,
        images_num,
    )

    process_images(
        test_volumes_list,
        test_segmentation_list,
        nii_volumes_path,
        nii_segmentation_path,
        png_test_volumes_path,
        png_test_segmentation_path,
        skip,
        images_num,
    )


if __name__ == "__main__":

    create_images = None
    skip = 1
    test_nii_files = 1
    data_dir = "data"
    data_type_dir = "CT"
    ct_images_directory = "nii_data"
    images_directory = "png"
    train_dir = "train_dir"
    test_dir = "test_dir"
    volumes_dir = "data"
    segmentation_dir = "mask"

    nii_directory = os.path.join(data_dir, data_type_dir, ct_images_directory)
    png_directory = os.path.join(data_dir, data_type_dir, images_directory)

    create_sets(
        create_images,
        skip,
        test_nii_files,
        nii_directory,
        png_directory,
        train_dir,
        test_dir,
        volumes_dir,
        segmentation_dir,
    )
