import tensorflow as tf

"""

TODO:
    1. implement summaries for
        i. parameters
        ii. losses
        iii. gradients
    2. implement loss functions
        i. wasserstein

"""

class GAN(object):
    def _vanilla_loss_fn(
            self,
            g_z, 
            d_x,
            d_g_z):
        d_loss = tf.reduce_mean(tf.log(d_x)) + \
                tf.reduce_mean(tf.log(1-d_g_z))
        d_loss = -d_loss

        # non-saturating heuristic
        g_loss = tf.reduce_mean(tf.log(d_g_z))
        g_loss = -g_loss

        return g_loss, d_loss 


    def __init__(self, 
            generator, 
            discriminator, 
            loss_fn):
        self.generator = generator
        self.discriminator = discriminator
        if loss_fn == "vanilla":
            self.loss_fn = self._vanilla_loss_fn 


    def train(self, 
            sess,
            g_optimizer, 
            d_optimizer, 
            real_data, 
            config, 
            g_scope = "generator", 
            d_scope = "discriminator",
            hooks = None):
        # TODO: read about GraphDef and protobuf
        data_placeholder = tf.placeholder(real_data.dtype, real_data.shape)
        dataset = tf.data.Dataset.from_tensor_slices((real_data))
        dataset = dataset.shuffle(buffer_size = real_data.shape[0])
        #dataset = dataset.shuffle(buffer_size = 10000)
        dataset = dataset.batch(config.x_batch_size)
        iterator = dataset.make_initializable_iterator()

        x = iterator.get_next()
        z = tf.random_normal([config.z_batch_size, \
                config.z_dims], stddev = config.z_std)

        g_z = self.generator(z)
        d_x = self.discriminator(x)
        d_g_z = self.discriminator(g_z)

        g_loss, d_loss = self.loss_fn(g_z, d_x, d_g_z)         

        d_train_step = d_optimizer.minimize(d_loss, var_list = \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 
                    scope = d_scope))
        g_train_step = g_optimizer.minimize(g_loss, var_list = \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 
                    scope = g_scope))

        init = tf.global_variables_initializer()
        sess.run(init)

        batch_counter = 0
        for epoch in range(config.num_epochs):
            sess.run(iterator.initializer, feed_dict = {data_placeholder: real_data})
            print "epoch %d"%epoch
            while True:
                try:
                    _, _d_x = sess.run([d_train_step, d_x])
                    _, _g_z, _d_g_z = sess.run([g_train_step, g_z, d_g_z])
                    
                    if hooks != None:
                        for hook in hooks:
                            if batch_counter % hook.n == 0:
                                hook.f(**{"g_z": _g_z, 
                                    "d_x": _d_x,
                                    "d_g_z": _d_g_z,
                                    "epoch": epoch})

                except tf.errors.OutOfRangeError:
                    break
                
                batch_counter += 1

