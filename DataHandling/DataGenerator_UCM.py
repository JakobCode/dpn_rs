import numpy as np
from PIL import Image
import os

ucm_classes_whole = {"agricultural": 0,
                     "airplane": 1,
                     "baseballdiamond": 2,
                     "beach": 3,
                     "buildings": 4,
                     "chaparral": 5,
                     "denseresidential": 6,
                     "forest": 7,
                     "freeway": 8,
                     "golfcourse": 9,
                     "harbor": 10,
                     "intersection": 11,
                     "mediumresidential": 12,
                     "mobilehomepark": 13,
                     "overpass": 14,
                     "parkinglot": 15,
                     "river": 16,
                     "runway": 17,
                     "sparseresidential": 18,
                     "storagetanks": 19,
                     "tenniscourt": 20
                     }

_ucm_class_splits = [[
                    ["forest",
                     "harbor",
                     "river",
                     "agricultural",
                     "beach",
                     "golfcourse",
                     "chaparral"
                     ],
                     ["airplane",
                      "baseballdiamond",
                      "intersection",
                      "mediumresidential",
                      "mobilehomepark",
                      "buildings",
                      "freeway"],

                     ["overpass",
                      "parkinglot",
                      "runway",
                      "sparseresidential",
                      "storagetanks",
                      "tenniscourt",
                      "denseresidential"
                      ]]]

def get_ucm_class_splits(rand=False, seed=42):
    
    if not rand:
        return _ucm_class_splits[0]
    else:
        np.random.seed(seed)
        num_classes = len(ucm_classes_whole)
        id_list = np.arange(num_classes)
        np.random.shuffle(id_list)

        return [np.array(list(ucm_classes_whole))[id_list[:(num_classes // 3)]],
               np.array(list(ucm_classes_whole))[id_list[(num_classes // 3):-(num_classes // 3)]],
               np.array(list(ucm_classes_whole))[id_list[-(num_classes // 3):]]]


def generator_ucm(root_folder,
                  batch_size=32,
                  filter_classes=None,
                  band_filter=None,
                  set_fraction=None,
                  seed=10):
    """
    root_folder                 path to root folder of dataset                      string
    batch_size                  number of samples per generated batches             integer
    filter_classes              function for filtering out classes                  function
    set_fraction                fraction that should be sampled from data set       float
    seed                        seed value for reproducability                      int
    """
    ucm_classes = ucm_classes_whole.copy()

    # filter out sets if needed
    if filter_classes is not None:
        for class_name in ucm_classes_whole:
            if class_name not in filter_classes:
                del ucm_classes[class_name]

    for num, key in enumerate(ucm_classes.keys()):
        ucm_classes[key] = num

    # create list of folders containing class images
    class_folders = [os.path.join(root_folder, class_name) for class_name in ucm_classes.keys()]
    assert all([os.path.isdir(class_path) for class_path in class_folders])

    num_classes = len(class_folders)

    # Load complete data into memory
    class_imgs_dict = {}
    class_labels_dict = {}
    np.random.seed(seed=seed)

    if band_filter is not None:
        np.sort(band_filter)
    else:
        band_filter = [0,1,2]

    for class_name, dirname in zip(ucm_classes.keys(), class_folders):
        class_imgs_dict[class_name] = []

        fnames = os.listdir(dirname)

        # sample fraction from complete data set
        if set_fraction is not None:
            indices = np.arange(len(fnames))
            np.random.shuffle(indices)
            start = int(set_fraction[0] * len(indices))
            end = int(set_fraction[1] * len(indices))
            indices = indices[start:end]
            fnames = np.array(fnames)[indices]

        for fname in fnames:
            im = Image.open(os.path.join(dirname, fname))
            if im.size[0] != 256 or im.size[1] != 256:
                im = im.resize([256, 256], Image.ANTIALIAS)

            imarray = np.array(im)[:,:,band_filter]


            class_imgs_dict[class_name].append(imarray)

        label = np.zeros(num_classes)
        label[ucm_classes[class_name]] = 1
        class_labels_dict[class_name] = np.repeat([label], len(class_imgs_dict[class_name]), axis=0)

    class_imgs = np.concatenate([class_imgs_dict[key] for key in class_imgs_dict.keys()], axis=0)
    class_labels = np.concatenate([class_labels_dict[key] for key in class_labels_dict.keys()], axis=0)
    indices = np.arange(len(class_imgs))

    # actual generator function
    def f():

        np.random.seed(seed)

        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):

                if i + batch_size > len(indices):
                    break

                # build next batch
                batch_indices = indices[i:i + batch_size]
                by = class_labels[batch_indices, :]
                bx = class_imgs[batch_indices, :]

                yield bx, by

    # return generator and number of iterations
    return f, len(indices) // batch_size


