#! /usr/local/bin/python3
import argparse
import Train
import Inference

def train(dataset_path):
    print('here')
    train_object = Train({
        'dataset_path': dataset_path
    })
    train_object.start()

def inference(image_type = 'TWI'):
    print('here')
    inference_object = Inference()
    inference_object.set_image_type(image_type)
    inference_object.start()

def main():
    parser = argparse.ArgumentParser(description="GAN Prostate Cancer augmentation using DCGAN")

    parser.add_argument("operation", choices=["train", "inference"], required=True, help="The operation to perform")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset with shape (N, 160,160, 3)")
    parser.add_argument("--image_type", type=str, help="The image type of the model (ADC, TFE, TWI)")

    args = parser.parse_args()

    if(args.operation == "train"):
        if (args.dataset_path == None):
            raise Exception('Dataset is required')
        train(dataset_path=args.dataset_path)
    elif(args.operation == "inference"):
        inference(args.image_type)
    else:
        raise Exception('Invalid Operation')

if __name__ == "__main__":
    main()



