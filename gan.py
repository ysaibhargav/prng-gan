import tensorflow as tf

"""

TODO:
    1. implement summaries for
        i. parameters
        ii. losses
        iii. gradients

"""

class GAN(object):
    def _vanilla_loss_fn(
            self,
            z, 
            x):
        g_z = self.generator(z)
        d_x = self.discriminator(x)
        d_g_z = self.discriminator(g_z)

        d_loss = tf.reduce_mean(tf.log(d_x)) + \
                tf.reduce_mean(tf.log(1-d_g_z))
        d_loss = -d_loss

        # non-saturating heuristic
        g_loss = tf.reduce_mean(tf.log(d_g_z))
        g_loss = -g_loss

        # TODO: namedtuple for return vals
        return g_loss, d_loss, [g_z, d_x, d_g_z] 


    def _wasserstein_loss_fn(
            self,
            z,
            x,
            use_gp=True,
            lmbda=10):
        g_z = self.generator(z)
        d_x = self.discriminator(x)
        d_g_z = self.discriminator(g_z)

        g_loss = -tf.reduce_mean(d_g_z)
        d_loss = tf.reduce_mean(d_g_z) - tf.reduce_mean(d_x)

        # gradient penalty 
        if use_gp:
            epsilon = tf.random_uniform(
                    shape=[int(z.shape[0]), 1],
                    minval=0.,
                    maxval=1.)

            x_hat = x + epsilon*(g_z - x)
            d_x_hat = self.discriminator(x_hat)
            grad_d_x_hat = tf.gradients(d_x_hat, [x_hat])[0]
            EPS = 1e-8
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_d_x_hat), \
                    reduction_indices=1)+EPS)
            gp = tf.reduce_mean((1.-slopes)**2)

            d_loss += lmbda * gp

        return g_loss, d_loss, [g_z, d_x, d_g_z]


    def __init__(self, 
            generator, 
            discriminator, 
            loss_fn):
        self.generator = generator
        self.discriminator = discriminator
        self._loss_fn = loss_fn


    def _data_handler(self,
            config,
            real_data):
        round_sz = config.x_batch_size*(real_data.shape[0]\
                //config.x_batch_size)
        real_data = real_data[:round_sz]

        dataset = tf.data.Dataset.from_tensor_slices((real_data))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(config.x_batch_size)
        iterator = dataset.make_initializable_iterator()

        x = iterator.get_next()
        z = tf.random_normal([config.z_batch_size, \
                config.z_dims], stddev = config.z_std)

        return x, z, iterator


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
        x, z, iterator = self._data_handler(config, real_data)

        if self._loss_fn == "vanilla":
            self.loss_fn = self._vanilla_loss_fn 
            self.n_critic = 1
        elif self._loss_fn == "wasserstein":
            self.loss_fn = self._wasserstein_loss_fn
            self.n_critic = config.n_critic

        g_loss, d_loss, vals = self.loss_fn(z, x)
        g_z, d_x, d_g_z = vals

        # TODO: what's colocate_gradients_with_op?
        d_grad_var = d_optimizer.compute_gradients(d_loss, var_list = \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 
                    scope = d_scope))
    
        d_train_step = d_optimizer.apply_gradients(d_grad_var, \
                name = "D_loss_minimize")

        g_grad_var = g_optimizer.compute_gradients(g_loss, var_list = \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 
                    scope = g_scope))

        g_train_step = g_optimizer.apply_gradients(g_grad_var, \
                name = "G_loss_minimize")

        init = tf.global_variables_initializer()
        sess.run(init)

        batch_counter = 0
        for epoch in range(config.num_epochs):
            sess.run(iterator.initializer)

            print "epoch %d"%epoch
            while True:
                try:
                    for _n_critic in range(self.n_critic):
                        _, _d_x = sess.run([d_train_step, d_x])
                    _, _g_z, _d_g_z = sess.run([g_train_step, g_z, d_g_z])
                    
                except tf.errors.OutOfRangeError:
                    break
                
                batch_counter += self.n_critic

            if hooks != None:
                for hook in hooks:
                    if epoch % hook.n == 0:
                        hook.f(**{"g_z": _g_z, 
                            "d_x": _d_x,
                            "d_g_z": _d_g_z,
                            "epoch": epoch})

