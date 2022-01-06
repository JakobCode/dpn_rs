import tensorflow as tf

def build_in_out_generator(gen_in,
                           input_shape,
                           output_shape,
                           gen_ood=None,
                           batch_size=32,
                           crop_shape=None,
                           resize_shape=None,
                           vertical_flip=True,
                           horizontal_flip=True,
                           norm_mu=0,
                           norm_std=1,
                           beta_in=1,
                           beta_out_out=0,
                           beta_out_in=0):

    num_classes = output_shape[-1]

    # no out-of-distribution set to consider
    if gen_ood is None:
        flow = gen_in
    else:
        batch_size = 2 * batch_size
        gen_in = gen_in()
        gen_ood = gen_ood()

        def flow_data():
            while True:
                x0, y0 = next(gen_in)
                y0 = y0 * (beta_in-1) + tf.ones_like(y0) * beta_out_in
                x1, y1 = next(gen_ood)
                y1 = tf.ones_like(y0) * beta_out_out
                xf = tf.concat((x0, x1), axis=0)
                yf = tf.concat((y0, y1), axis=0)

                yield xf, yf
        flow = flow_data

    

    flow_ds = tf.data.Dataset.from_generator(flow,
                                             (tf.float32, tf.float32),
                                             output_shapes=((tf.TensorShape([batch_size, *input_shape]),
                                                             tf.TensorShape([batch_size, *output_shape]))))

    # apply transformations on dataset
    if input_shape[-1] == 1:
        flow_ds.map(lambda x, y: [tf.concat([x, x, x], -1), y])
    if crop_shape is not None:
        flow_ds = flow_ds.map(lambda x, y: [tf.image.random_crop(x, [batch_size,*crop_shape]), y])
    if resize_shape is not None:
        flow_ds = flow_ds.map(lambda x, y: [tf.image.resize(x, resize_shape[:2]), y])
    if vertical_flip:
        flow_ds = flow_ds.map(lambda x, y: [tf.image.random_flip_up_down(x), y])
    if horizontal_flip:
        flow_ds = flow_ds.map(lambda x, y: [tf.image.random_flip_left_right(x), y])

    flow_ds = flow_ds.map(lambda x, y: [x-norm_mu,  y])
    flow_ds = flow_ds.map(lambda x, y: [x/norm_std, y])

    if input_shape[-1] == 1:
        flow_ds = flow_ds.map(lambda x, y: [tf.concat([x, x, x], axis=-1), y])

    return flow_ds.prefetch(tf.data.experimental.AUTOTUNE)

