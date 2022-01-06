import os
import argparse

from settings import *
from DataHandling.DataGenerator import build_in_out_generator
from DataHandling.DataGenerator_UCM import generator_ucm, get_ucm_class_splits
from DataHandling.DataGenerator_AID import generator_aid, get_aid_class_splits
from Models.model_prepare import prepare_model

import random
import tensorflow as tf
import numpy as np

def run_experiment(data_set, approach,  exp_save_path, seed):

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    batch_size = 32
    model_type = Models.ResNet50

    mu, std = 0, 255

    training_fraction = [0.0, 0.7]
    validation_fraction = [0.7, 1.0]

    num_epochs = 25
    band_filter_train_in = band_filter_train_ood = band_filter_val_in = band_filter_val_ood = None

    if data_set is Dataset.UCM:
        data_root_path = ucm_root_path
        input_shape = [256, 256, 3]
        crop_shape = [241, 241, 3]
        resize_shape = [256, 256, 3]       
        generator = generator_ucm
        classes_in, classes_out_training, _ = get_ucm_class_splits()

    elif data_set is Dataset.AID:
        data_root_path = aid_root_path
        input_shape = [600, 600, 3]
        resize_shape = [256, 256, 3]
        crop_shape = [500, 500, 3]        
        generator = generator_aid
        classes_in, classes_out_training, _ = get_aid_class_splits()

    num_classes = len(classes_in)

    if approach is Approaches.dpn_rs or approach is Approaches.dpn_plus:
        beta_in=2
        beta_out_in=0
        beta_out_out=1.0 / num_classes
    elif approach is Approaches.prior_kl_forward or approach is Approaches.prior_kl_reverse:
        beta_in=100
        beta_out_in=1
        beta_out_out=1
    elif approach is Approaches.enn_cross_entropy:
        beta_in=2
        beta_out_in=0
        beta_out_out=None

    # Create data Loader
    gen_in_train, gen_in_train_steps = generator(root_folder=data_root_path,
                                                        batch_size=batch_size,
                                                        set_fraction=training_fraction,
                                                        filter_classes=classes_in,
                                                        band_filter=band_filter_train_in,
                                                        seed=seed)

    if approach is Approaches.enn_cross_entropy:
        gen_out_train = None
    else:
        gen_out_train, _ = generator(root_folder=data_root_path,
                                    batch_size=batch_size,
                                    set_fraction=training_fraction,
                                    filter_classes=classes_out_training,
                                    band_filter=band_filter_train_ood,
                                    seed=seed)


    train_flow = build_in_out_generator(gen_in=gen_in_train,
                                      gen_ood=gen_out_train,
                                      batch_size=batch_size,
                                      input_shape=input_shape,
                                      crop_shape=crop_shape,
                                      resize_shape=resize_shape,
                                      vertical_flip=True,
                                      horizontal_flip=True,
                                      norm_mu=mu,
                                      norm_std=std,
                                      output_shape=[num_classes],
                                      beta_in=beta_in,
                                      beta_out_in=beta_out_in,
                                      beta_out_out=beta_out_out)
                                      
    gen_in_val, gen_in_val_steps = generator(root_folder=data_root_path,
                                            batch_size=batch_size,
                                            set_fraction=validation_fraction,
                                            filter_classes=classes_in,
                                            band_filter=band_filter_val_in,
                                            seed=seed)


    if approach is Approaches.enn_cross_entropy:
        gen_out_val = None
    else:
        gen_out_val, _ = generator(root_folder=data_root_path,
                                    batch_size=batch_size,
                                    set_fraction=validation_fraction,
                                    filter_classes=classes_out_training,
                                    band_filter=band_filter_val_ood,
                                    seed=seed)

    val_flow = build_in_out_generator(gen_in=gen_in_val,
                                      gen_ood=gen_out_val,
                                      batch_size=batch_size,
                                      input_shape=input_shape,
                                      crop_shape=crop_shape,
                                      resize_shape=resize_shape,
                                      vertical_flip=True,
                                      horizontal_flip=True,
                                      norm_mu=mu,
                                      norm_std=std,
                                      output_shape=[num_classes],
                                      beta_in=beta_in,
                                      beta_out_in=beta_out_in,
                                      beta_out_out=beta_out_out)


    # Build model and callbacks
    model, callbacks = prepare_model(model_type=model_type,
                                    approach=approach,
                                    num_classes=num_classes,
                                    input_shape=resize_shape)
    model.summary()

    # Start training process
    model.fit(train_flow,
                steps_per_epoch=gen_in_train_steps,
                validation_data=val_flow,
                validation_steps=gen_in_val_steps,
                epochs=num_epochs,
                max_queue_size=100,
                callbacks=[callbacks],
                verbose=1)

    model.save_weights(os.path.join(exp_save_path, "final_model"))


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description='Foo')
    parser.add_argument('-d','--data', help='Name of data set. [ucm, aid]', type=str, required=True)
    parser.add_argument('-a', '--approach', help=f"Approach use for ood detection. {['dpn_rs', 'prior_forward', 'prior_reverse', 'dpn_plus', 'evidential_cross_entropy']}", type=str, default="dpn_rs")
    parser.add_argument('-s','--seed', help='Output file name.', type=int, default=42)
    parser.add_argument('-p', '--path', help='Path for saving results.', type=str, default='./')
    args = parser.parse_args()


    assert args.approach in ["dpn_rs", "prior_kl_forward", "prior_kl_reverse", "dpn_plus", "evidential_cross_entropy"], f"approach '{args.approach}' not valid argument!"
    
    if args.approach == "dpn_rs":
        approach = Approaches.dpn_rs
    elif args.approach == "prior_kl_forward":
        approach = Approaches.prior_kl_forward
    elif args.approach == "prior_kl_reverse":
        approach = Approaches.prior_kl_reverse
    elif args.approach == "dpn_plus":
        approach = Approaches.dpn_plus
    elif args.approach == "evidential_cross_entropy":
        approach = Approaches.enn_cross_entropy

    assert args.data in ["ucm", "aid"], f"Data set '{args.data}' is not a valid option (aid / ucm)."
    if args.data == "ucm": 
        data_set = Dataset.UCM
    elif args.data == "aid":
        data_set = Dataset.AID

    save_path = args.path
    seed = args.seed

    run_experiment(data_set=data_set, approach=approach, exp_save_path=save_path, seed=seed)