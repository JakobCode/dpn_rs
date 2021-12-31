from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from Experiments.settings import Models
import os


def resnet(input_shape, num_classes, model_type=Models.ResNet50, weights='imagenet', include_top=False):

    inputs = Input(input_shape)
    input_shape_s = "_" + str(input_shape).replace("[","").replace("]","").replace(" ", "_").replace(",","")
    if model_type is Models.ResNet50:
        if os.path.exists("Models/ResNet/ResNet50" + str(input_shape_s)):
            resnet_base = load_model("Models/ResNet/ResNet50" + str(input_shape_s))

        else:
            resnet_base = ResNet50V2(include_top=include_top,
                              weights=weights,
                              input_tensor=inputs,
                              input_shape=None,
                              pooling=None,
                              #classes=num_classes
                          )
            resnet_base.save("Models/ResNet/ResNet50" + str(input_shape_s))

    elif model_type is Models.ResNet101:
        if os.path.exists("Models/ResNet/ResNet101" + str(input_shape_s)):
            resnet_base = load_model("Models/ResNet/ResNet101" + str(input_shape_s))
        else:
            resnet_base = ResNet101V2(include_top=include_top,
                              weights=weights,
                              input_tensor=inputs,
                              input_shape=None,
                              pooling=None,
                              #classes=num_classes
                              )
            resnet_base.save("Models/ResNet/ResNet101" + str(input_shape_s))

    elif model_type is Models.ResNet152:
        if os.path.exists("Models/ResNet/ResNet152" + str(input_shape_s)):
            resnet_base = load_model("Models/ResNet/ResNet152" + str(input_shape_s))
        else:
            resnet_base = ResNet152V2(include_top=include_top,
                              weights=weights,
                              input_tensor=inputs,
                              input_shape=None,
                              pooling=None,
                              #classes=num_classes
                              )
            resnet_base.save("Models/ResNet/ResNet152" + str(input_shape_s))

    else:
        raise Exception("Resnet Type not known")

    f1 = Flatten()(resnet_base.output)

    num_classes = num_classes if num_classes > 2 else 1
    d1 = Dense(num_classes, name="dense_3")(f1)

    resnet_model = Model(inputs=resnet_base.inputs, outputs=d1)

    return resnet_model


if __name__=="__main__":

    test = resnet(inputs=Input([225,225,3]),
                 num_classes=10,
                 model_type=101)

    test.summary()


