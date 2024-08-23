#!/usr/bin/env python3

import argparse
import Train
import Inference

def train(**kwargs):
    train_object = Train(kwargs)
    train_object.start()

def inference(image_type = 'TWI'):
    inference_object = Inference()
    inference_object.set_image_type(image_type)
    inference_object.start()

def main():
    parser = argparse.ArgumentParser(description="GAN Prostate Cancer augmentation using DCGAN")

    parser.add_argument("operation", choices=["train", "inference"],help="The operation to perform")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset with shape (N, 160,160, 3)")
    parser.add_argument("--gen_optimizer", type=str, help="The optimizer to use in training (default is Adam)", default='Adam')
    parser.add_argument("--gen_learning_rate", type=str, help="The learning rate to use in training the generator (default is 2e-3)", default=2e-3)
    parser.add_argument("--gen_beta_1", type=str, help="The first momentum term for the generator (default is 0.5)", default=0.5)
    parser.add_argument("--gen_beta_2", type=str, help="The first momentum term for the generator (default is 0.9)", default=0.9)
    parser.add_argument("--disc_optimizer", type=str, help="The optimizer to use in training (default is Adam)", default='Adam')
    parser.add_argument("--disc_learning_rate", type=str, help="The learning rate to use in training the discriminator (default is 2e-3)", default=2e-3)
    parser.add_argument("--disc_beta_1", type=str, help="The first momentum term for the discriminator  (default is 0.5)", default=0.5)
    parser.add_argument("--disc_beta_2", type=str, help="The first momentum term for the discriminator (default is 0.9)", default=0.9)
    
    parser.add_argument("--epochs", type=str, help="The number of epochs", default=3000)
    parser.add_argument("--image_type", type=str, help="The image type of the model (ADC, TFE, TWI)", default='TWI')

    args = parser.parse_args()

    if(args.operation == "train"):
        if (args.dataset_path == None):
            raise Exception('Dataset is required')
        params = {
            'dataset_path': args.dataset_path,
            'epochs': args.epochs,
            'gen_optimizer': args.gen_optimizer,
            'gen_learning_rate': args.gen_learning_rate,
            'gen_beta_1': args.gen_beta_1,
            'gen_beta_2': args.gen_beta_2,
            'disc_optimizer': args.disc_optimizer,
            'disc_learning_rate': args.disc_learning_rate,
            'disc_beta_1': args.disc_beta_1,
            'disc_beta_2': args.disc_beta_2
        }
        train( **params )
    elif(args.operation == "inference"):
        inference(args.image_type)
    else:
        raise Exception('Invalid Operation')

if __name__ == "__main__":
    main()



