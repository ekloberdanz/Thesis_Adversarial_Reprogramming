import tensorflow as tf
import numpy as np


# FGSM
class FGSM(object):
    def __init__(self, model, epsilon,  dataset='CIFAR'):
        self.model = model
        self.epsilon = epsilon

        if dataset == 'CIFAR':
            self.height = 32
            self.width = 32
            self.channel = 3
        elif dataset == 'MNIST':
            self.height = 28
            self.width = 28
            self.channel = 1
        self.xs = tf.Variable(np.zeros((1, self.height, self.width, self.channel), dtype=np.float32), name='modifier')
        self.xs_place = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.xs_orig = tf.Variable(np.zeros((1, self.height, self.width, self.channel), dtype=np.float32), name='original')
        self.ys = tf.placeholder(tf.int32, [None])
        self.y_variable = tf.Variable(np.zeros((1,), dtype=np.int32), name='target')
        # assign operations
        self.assign_x = tf.assign(self.xs, self.xs_place)
        self.assign_x_orig = tf.assign(self.xs_orig, self.xs_place)
        self.assign_y = tf.assign(self.y_variable, self.ys)
        # clip operation
        self.do_clip_xs = tf.clip_by_value(self.xs, 0, 1)
        # logits
        self.logits = model(self.xs)
        # loss
        y_one_hot = tf.one_hot(self.y_variable, 10)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits , labels=y_one_hot)
        loss = tf.reduce_mean(loss)
        # Define gradient of loss wrt input
        self.grad = tf.gradients(loss, self.xs)[0]
        self.grad_sign = tf.sign(self.grad)

    def perturb(self, x, y, sess):
        # initialize
        sess.run(self.xs.initializer)
        sess.run(self.y_variable.initializer)
        sess.run(self.xs_orig.initializer)
        # assign
        sess.run(self.assign_x, feed_dict={self.xs_place: x})
        # sess.run(self.assign_x_orig, feed_dict={self.xs_orig: x})
        sess.run(self.assign_y, feed_dict={self.ys: y})

        # generate adv example
        #grad = sess.run(self.grad)
        grad_sign = sess.run(self.grad_sign)
        adv_x = x - self.epsilon * grad_sign

        return adv_x

    def sample_gradient(self, x, y, sess, gradient_samples):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
        for grad_idx in range(gradient_samples):
            grad = sess.run(self.grad, feed_dict={self.ys: y})
            grad_list[grad_idx, :, :, :, :] = grad

        return grad_list

    def perturb_gm(self, x, y, sess, gradient_samples):
        # initialize
        sess.run(self.xs.initializer)
        sess.run(self.y_variable.initializer)
        sess.run(self.xs_orig.initializer)
        # assign
        sess.run(self.assign_x, feed_dict={self.xs_place: x})
        # sess.run(self.assign_x_orig, feed_dict={self.xs_place: x})
        sess.run(self.assign_y, feed_dict={self.ys: y})

        grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
        for i in range(gradient_samples):
            grad = sess.run(self.grad)
            grad_list[i, :, :, :, :] = grad
        grad_mean = np.mean(grad_list, axis=0, keepdims=False)
        grad_mean_sign = np.sign(grad_mean)
        adv_x = x - self.epsilon * grad_mean_sign

        return adv_x
