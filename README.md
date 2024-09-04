## MRI Image Augmentation and Training Tool

This project provides a tool for training a model to generate augmented MRI images, specifically focused on prostate cancer (PCa MRI).

### Install packages
To install the compatible packages for the scripts, run
```
pip install -r requirements.txt
```
### Training

The `main.py` script simplifies the training process with the following features:

* **Entry point:** It acts as the main entry point for training.
* **Parameter control:** Accepts various training parameters for customization, including:
    * `dataset_path`: Path to your MRI dataset containing normalized samples with dimension 160x160 (required)
    * `epochs`: Number of training epochs (default: 3000)
    * `gen_learning_rate`: Learning rate for the generator (default: 2e-3)
    * `disc_learning_rate`: Learning rate for the discriminator (default: 2e-3)
    * `gen_beta_1`: First momentum term for the generator optimizer (default: 0.5)
    * `disc_beta_1`: First momentum term for the discriminator optimizer (default: 0.5)
    * `gen_beta_2`: Second momentum term for the generator optimizer (default: 0.9)
    * `disc_beta_2`: Second momentum term for the discriminator optimizer (default: 0.999)

**Example:**

To train with 1000 epochs, run:

```
./main.py training --dataset_path {PATH_TO_DATASET} --epochs 1000 --gen_learning_rate 2e-3 --gen_beta_1 0.8 --disc_beta_1 0.8 --gen_beta_2 0.99 --disc_beta_2 0.999
```

### Augmentation

The `main.py` script also facilitates generating augmented datasets:

* **Simple workflow:** Requires only the `image_type` parameter to specify the MRI image type.
* **Accessibility:** Designed to be easy to use, even for users with limited model inference experience.

**Example:**

To generate T2-weighted MRI augmentations (TWI), run:

```
./main.py inference --image_type TWI
```

**Explanation:**

* `image_type`: Specifies the type of PCa MRI image to generate. Defaults to "TWI" for T2-weighted imaging. You can adjust this parameter based on your dataset.

Samples of generated images:
- Augmented T2-Weighted Images - 1 sequence with 3 mid layers
  <br>
  <img width="147" alt="image" src="https://github.com/user-attachments/assets/a2802f2e-f188-448f-90c4-aaebb88c39f5">
  <img width="147" alt="image" src="https://github.com/user-attachments/assets/38d7890f-e199-4f26-835d-ae122422aecb">
  <img width="147" alt="image" src="https://github.com/user-attachments/assets/a34be1c7-300b-4b12-a63d-c3cc76133f98">




