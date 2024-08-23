import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import date, datetime
import make_generator_model
import make_discriminator_model
import TrainingUtils
import Eval

class Train:
    NOISE_DIM = 400

    today = (date.today()).strftime("%d-%m-%Y")
    timestamp = int(datetime.now().timestamp())

    _checkpoint_dir = './training_checkpoints'
    _img_dir = './AugmentedImages'
    _save_gen_dir = None
    _save_disc_dir = None
    _save_fid_dir = None
    _save_gen_loss_dir = None
    _save_disc_loss_dir = None
    _augmented_images_dir = None
    _augmented_image_preview_dir = None
    _save_seed_dir = None

    _utils = None
    _id = None
    _num_of_samples = 30
    _epochs = 100
    _seed = None
    _batch_size = 30
    _fid_scores = []
    _generator_loss_arr = []
    _discriminator_loss_arr = []
    _train_dataset = None

    def __init__(self, params) -> None:
        if (params.get('dataset_path') == None):
            raise Exception("Dataset is required")
        utils_params = {
            'dataset_path': params.get('dataset_path')
        }
        if (params.get('seed') != None):
            utils_params['seed'] = params.get('seed')
        else:
            utils_params['seed'] = np.random.normal(0,1,size=[self.get_number_of_samples(), self.NOISE_DIM])
        if (params.get('opt') != None):
            utils_params['opt'] = params.get('opt')
        if (params.get('crossEntropy') != None):
            utils_params['crossEntropy'] = params.get('crossEntropy')
        self._utils = TrainingUtils(**utils_params)
        if (params.get('id') != None):
            self.set_id(params.get('id'))
        else:
            self.set_id(self._utils.get_random_id())
        if (params.get('checkpoint_dir') != None):
            self.set_checkpoint_dir(params.get('checkpoint_dir'))
        if (params.get('num_of_samples')):
            self.set_number_of_samples(params.get('num_of_samples'))
        if (params.get('batch_size')):
            self.set_batch_size(params.get('batch_size'))
        if (params.get('epochs')):
            self.set_epochs(params.get('epochs'))
        self._set_training_data_dirs()
        self._train_dataset = self._get_dataset_tensor_array()

    def get_batch_size(self):
        return self._batch_size
    def set_batch_size(self, size):
        self._batch_size = size
    def get_id(self):
        return self._id
    def set_id(self, id):
        self._id = id
    def get_checkpoint_dir(self):
        return self._checkpoint_dir
    def set_checkpoint_dir(self, dir):
        self._checkpoint_dir = dir;
    def get_number_of_samples(self):
        return self._num_of_samples
    def set_number_of_samples(self, n):
        self._num_of_samples = n
    def get_epochs(self):
        return self._epochs
    def set_epochs(self, n):
        self._epochs = n
    
    def start(self):
        generator = self._create_generator_model()
        discriminator = self._create_discriminator_model()
        generator_optimizer = self._get_optimizer(opt='Adam', learning_rate=2e-3)
        discriminator_optimizer = self._get_optimizer(opt='Adam', learning_rate=2e-3)
        self._train(self._train_dataset, self.get_epochs(),generator=generator, discriminator=discriminator, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
        self.end(generator=generator, discriminator=discriminator)

    def end(self, generator, discriminator):
        self._save_metrics()
        self._plot_model_metric_graphs()
        self._plot_FID_graph()
        self._generate_and_save_augmented_images(generator=generator)
        self._save_model_files(generator=generator, discriminator=discriminator)

    def _append_fid_score(self, fid):
        self._fid_scores.append(fid)

    def _append_gen_loss(self, loss):
        self._generator_loss_arr.append(loss)
    
    def _append_disc_loss(self, loss):
        self._discriminator_loss_arr.append(loss)

    def _set_training_data_dirs(self):
        self._save_gen_dir = './gen-model-{}-{}'.format(self.today, self.get_id())
        self._save_disc_dir = './discriminator-model-{}-{}'.format(self.today, self.get_id())
        self._save_fid_dir = './fid_{}_{}_{}'.format(self.today, self.get_id(), self.timestamp)
        self._save_gen_loss_dir = './generator_loss_{}_{}_{}'.format(self.today, self.get_id(), self.timestamp)
        self._save_disc_loss_dir = './discriminator_loss_{}_{}_{}'.format(self.today, self.get_id(), self.timestamp)
        self._augmented_images_dir = './AugmentedImages/{}-{}-gan_images/'.format(self.today, self.get_id())
        self._augmented_image_preview_dir = './ProstateX/image_preview'
        self._save_seed_dir = './seed_{}_{}_{}'.format(self.today, self.get_id(), self.timestamp)

    def _create_generator_model(self):
        return make_generator_model()

    def _create_discriminator_model(self):
        return make_discriminator_model()
    
    def _get_dataset_tensor_array(self):
        ds = self._utils.get_training_dataset()
        return tf.data.Dataset.from_tensor_slices(ds).shuffle(ds.shape[0]).batch(self.get_batch_size())
    
    def _get_optimizer(self, opt, **kwargs):
        return self._utils.get_optimizer(opt, **kwargs)

    
    def _view_images(self, lists, date="", epoch=0, run_id="", save = False):
        fig, axes = plt.subplots(nrows=1, ncols=lists.shape[3], figsize=(20, 20), squeeze=False)
        plt.subplots_adjust(wspace=10, hspace=10)
        fig.tight_layout()
        for i, ax_row in enumerate(axes):
            for j, ax_col in enumerate(ax_row):
                sl = j;
                pred = lists[i]
                ax_col.imshow(pred[:,:,sl],cmap='gray')
                ax_col.axis('off')

                if(save):
                    image_dir = self._augmented_images_dir
                    file_name = '{}/image_at_epoch_{:04d}.png'.format(image_dir, epoch)
                    if run_id == "":
                        image_dir = self._augmented_image_preview_dir
                        file_name = '{}/image_{}.png'.format(image_dir,self._utils.get_random_id())

                    self._utils.save_plt_image(plt, image_dir, file_name)
        plt.show()

    def _generate_and_save_images(self, model, epoch, test_input, run_id, date, timestamp):
            predictions = model(test_input, training=False)
            self._view_images(predictions, date, epoch, run_id, True)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def _train_step(self, images, generator, discriminator, generator_optimizer, discriminator_optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(self._utils.get_seed(), training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = self._utils.compute_generator_loss(fake_output)
            disc_loss = self._utils.compute_discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return gen_loss, disc_loss

    def _compute_stats(self, generator, discriminator):
        real_data = self._train_dataset.take(1)
        real_data = next(iter(self._train_dataset))

        generated_images = generator(self._utils.get_seed(), training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = self._utils.compute_generator_loss(fake_output)
        disc_loss = self._utils.compute_discriminator_loss(real_output, fake_output)

        eval = Eval()
        casted_gen_data = tf.cast(generated_images, dtype=tf.float64)
        fid_score = eval.get_fid(real_data, casted_gen_data)
        print("fid score: {}".format(fid_score))

        self._append_fid_score(fid_score)
        self._append_gen_loss(tf.keras.backend.get_value(gen_loss))
        self._append_disc_loss(tf.keras.backend.get_value(disc_loss))

    def _train(self, dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
        checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator)
        overall_start = time.time()
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                self._train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

            if (epoch + 1) % 1000 == 0:
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-overall_start))
                print("saving checkpoint")
                checkpoint.save(file_prefix = checkpoint_prefix)
                self._generate_and_save_images(generator,
                                    epoch+1,
                                    self._utils.get_seed(), self.get_id(), self.today, self.timestamp)
                self._compute_stats(generator=generator, discriminator=discriminator)
                print("stats computed")
                overall_start = time.time()

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        self._generate_and_save_images(generator,
                                epochs,
                                self._utils.get_seed(), self.get_id(), self.today, self.timestamp)

    def _save_metrics(self):
        np.save(self._save_gen_loss_dir, self._generator_loss_arr)
        np.save(self._save_disc_loss_dir, self._discriminator_loss_arr)
        np.save(self._save_fid_dir, self._fid_scores)

    def _plot_model_metric_graphs(self):
        plt.title('Generator loss per step')
        plt.plot(self._generator_loss_arr, 'g--', self._discriminator_loss_arr, 'r--')
        plt.legend(['Generator Loss', 'Discriminator loss'], loc='upper left')
        self._utils.save_plt_image(plt, self._augmented_images_dir, '{}/{}'.format(self._augmented_images_dir, self.get_id()))

    def _plot_FID_graph(self):
        plt.title('Fid Scores')
        plt.plot(self._fid_scores, 'b--')
        self._utils.save_plt_image(plt, self._augmented_images_dir, '{}/fid-{}'.format(self._augmented_images_dir, self.get_id()))
        plt.figure()

    def _generate_and_save_augmented_images(self, generator):
        img_dir = '{}/{}'.format(self._img_dir, self.get_id())
        self._utils.save_images(generator, img_dir)

    def _save_model_files(self, generator, discriminator):
        generator.save("{}.keras".format(self._save_gen_dir))
        discriminator.save("{}.keras".format(self._save_disc_dir))
        seed = self._utils.get_seed()
        np.save(self._save_seed_dir, seed)

sys.modules[__name__] = Train
