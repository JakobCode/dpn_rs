import os
import argparse

from Experiments.settings import *
from DataHandling.DataGenerator import build_in_out_generator
from DataHandling.DataGenerator_UCM import generator_ucm, get_ucm_class_splits
from DataHandling.DataGenerator_AID import generator_aid, get_aid_class_splits
from Models.model_prepare import prepare_model

from Measures.Uncertainty_Measures import mutual_information, entropy, max_probability_from_logits, get_scores
import tensorflow as tf
import numpy as np

def run_experiment(data_set, approach, sava_path, seed):

    # SETTINGS
    exp_save_path = "./"

    batch_size = 32
    model_type = Models.ResNet50

    mu, std = 0, 255

    training_fraction = [0.0, 0.7]
    validation_fraction = [0.7, 1]
    test_fraction = [0.7, 1]

    num_epochs = 100
    band_filter_train_in = band_filter_train_ood = band_filter_val_in = band_filter_val_ood = None

    if data_set is Dataset.UCM:
        input_shape = [256, 256, 3]
        crop_shape = [241, 241, 3]
        resize_shape = [256, 256, 3]       
        generator = generator_ucm
        classes_in, classes_out_training, classes_out_testing = get_ucm_class_splits()

    elif data_set is Dataset.AID:
        input_shape = [600, 600, 3]
        resize_shape = [256, 256, 3]
        crop_shape = [500, 500, 3]        
        generator = generator_aid
        classes_in, classes_out_training, classes_out_testing = get_aid_class_splits()

    num_classes = len(classes_in)
    num_classes_out_training = len(classes_out_training)
    num_classes_out_testing = len(classes_out_testing)

    


    test_in_generator, test_steps_in = generator(root_folder=data_root_path,
                                                     batch_size=batch_size,
                                                     filter_classes=classes_in,
                                                     set_fraction=test_fraction,
                                                     seed=seed)

    test_out_generator, test_steps_out = generator(root_folder=data_root_path,
                                                       batch_size=batch_size,
                                                       filter_classes=classes_out_testing,
                                                       set_fraction=test_fraction,
                                                       seed=seed)

    test_steps = min(test_steps_in, test_steps_out)

    test_ds_in = tf.data.Dataset.from_generator(test_in_generator,
                                                (tf.float32, tf.float32),
                                                output_shapes=((tf.TensorShape([batch_size, *input_shape]),
                                                                tf.TensorShape([batch_size, num_classes]))))

    test_ds_in = test_ds_in.map(lambda x, y: [tf.image.resize_with_crop_or_pad(x, *crop_shape[:2]), y])
    test_ds_in = test_ds_in.map(lambda x, y: [tf.image.resize(x, resize_shape[:2]), y])
    test_ds_in = test_ds_in.map(lambda x, y: [(x-mu)/std, y])

    test_ds_in = test_ds_in.as_numpy_iterator()


    test_ds_out = tf.data.Dataset.from_generator(test_out_generator,
                                                 (tf.float32, tf.float32),
                                                 output_shapes=((tf.TensorShape([batch_size, *input_shape]),
                                                                 tf.TensorShape([batch_size, num_classes]))))

    test_ds_out = test_ds_out.map(lambda x, y: [tf.image.resize_with_crop_or_pad(x, *crop_shape[:2]), y])
    test_ds_out = test_ds_out.map(lambda x, y: [tf.image.resize(x, resize_shape[:2]), y])
    test_ds_out = test_ds_out.map(lambda x, y: [(x-mu)/std, y])

    test_ds_out = test_ds_out.as_numpy_iterator()

    # Build model and callbacks
    model, _ = prepare_model(model_type=model_type,
                             approach=approach,
                             num_classes=num_classes,
                             input_shape=resize_shape,
                             save_path=save_path,
                             saved_weights=None)

    model.load_weights(os.path.join(save_path, "final_model"))
    model.compile()

    y_pred_in = []
    y_true_in = []
    y_pred_out = []
    y_true_out = []

    for i in range(test_steps):
        print("Step %i of %i" %(i, test_steps))
        x, y = next(test_ds_out)
        y_true_out += [y]
        y_pred_out += [model.predict(x)]

        x, y = next(test_ds_in)
        y_true_in += [y]
        y_pred_in += [model.predict(x)]

    y_pred_in = np.array(tf.concat(y_pred_in, axis=0))
    y_pred_out = np.array(tf.concat(y_pred_out, axis=0))
    y_true_in = np.array(tf.concat(y_true_in, axis=0))
    y_true_out = np.array(tf.concat(y_true_out, axis=0))
    y_pred_in_one_hot = np.array(tf.one_hot(np.argmax(y_pred_in, -1), num_classes))
    y_pred_out_one_hot = np.array(tf.one_hot(np.argmax(y_pred_out, -1), num_classes))

    print("\n\n#############################################################################\n\n")
    print("performance samples")
    a = np.sum(y_true_in, axis=0)
    b = np.sum(y_true_in * y_pred_in_one_hot, axis=0)
    print("In Samples\n     Predictions: " + str(np.sum(y_pred_in_one_hot, axis=0)))
    print("     Groundtruth: " + str(a))
    print("     True Positives: " +str(b))
    print("     class accuracies in samples: " + str(b/a))

    a = np.sum(y_true_out, axis=0)
    b = np.sum(y_true_out * y_pred_out_one_hot, axis=0)
    print("Out Samples\n     Predictions: " + str(np.sum(y_pred_out_one_hot, axis=0)))
    print("     Groundtruth: " + str(a))
    print("     True Positives: " +str(b))
    print("     class accuracies out samples: " + str(b/a))
    print("")
    print("general scores: ")
    print(get_scores(y_pred_in, y_pred_out))
    print("\nclass wise mutual information - in")
    print([np.mean(mutual_information(y_pred_in[np.argmax(y_true_in, axis=-1) == c]), axis=-1) for c in range(num_classes)])
    print("\nclass wise max probability - in")
    print([np.mean(max_probability_from_logits(y_pred_in[np.argmax(y_true_in, axis=-1) == c]), axis=-1) for c in range(num_classes)])
    print("\nclass wise entropy - in")
    print([np.mean(entropy(y_pred_in[np.argmax(y_true_in, axis=-1) == c]), axis=-1) for c in range(num_classes)])
    print("\nclass wise mutual information - out")
    print([np.mean(mutual_information(y_pred_out[np.argmax(y_true_out, axis=-1) == c]), axis=-1) for c in range(num_classes_out_testing)])
    print("\nclass wise max probability - out")
    print([np.mean(max_probability_from_logits(y_pred_out[np.argmax(y_true_out, axis=-1) == c]), axis=-1) for c in range(num_classes_out_testing)])
    print("\nclass wise entropy - out")
    print([np.mean(entropy(y_pred_out[np.argmax(y_true_out, axis=-1) == c]), axis=-1) for c in range(num_classes_out_testing)])

    print("done")
    print("\n\n#############################################################################\n\n")


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description='Foo')
    parser.add_argument('-d','--data', help='Name of data set. [ucm, aid]', type=str, required=True)
    parser.add_argument('-a', '--approach', help=f"Approach use for ood detection. {['dpn_rs', 'prior_forward', 'prior_reverse', 'dpn_plus', 'evidential_cross_entropy']}", type=str, default="dpn_rs")
    parser.add_argument('-s','--seed', help='Output file name.', type=int, default=42)
    parser.add_argument('-p', '--path', help='Path for saving results.', type=str, default='./')
    args = parser.parse_args()


    assert args.approach in ["dpn_rs", "prior_forward", "prior_reverse", "dpn_plus", "evidential_cross_entropy"], f"approach '{args.approach}' not valid argument!"
    
    if args.approach == "dpn_rs":
        approach = Approaches.dpn_rs
    elif args.approach == "prior_forward":
        approach = Approaches.prior_kl_forward
    elif args.approach == "prior_reverse":
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

    run_experiment(data_set=data_set, approach=approach, sava_path=save_path, seed=seed)