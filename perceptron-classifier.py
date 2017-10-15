#!/usr/bin/env python

##########################################
#
# classifier.py: A Simple implementation of a Multilayer Perception network with TensorFlow 
#                used to classify the MNIST Dataset of handwritten digits (Source: http://yann.lecun.com/exdb/mnist/).
#
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 10/10/2017
#
# This file is distribuited under the terms of GNU General Public
#
########################################

from __future__ import print_function
import os
import tensorflow as tf

import argparse




FLAGS = None

def main():
    # Parameters
    learning_rate = FLAGS.learning_rate
    num_steps = FLAGS.steps
    batch_size = FLAGS.batch_size
    display_step = 100
    
    # Holy Hyperparameters
    num_layers = FLAGS.hidden_layer_num
    nodes_size = FLAGS.nodes_size 
    num_input = FLAGS.input_size
    num_classes = FLAGS.class_number
    save_path = FLAGS.model_file
 
   

    # Let's build the module 

    weights = [i for i in range(num_layers+1)]
    biases = [i for i in range(num_layers+1)]
    layers = [i for i in range(num_layers+1)]

    # Input Layer
    Input = tf.placeholder("float", [None, num_input])
    Labels = tf.placeholder("float", [None, num_classes])
    

    # Dinamically define our network

    # Input Layer
    weights[0] = tf.Variable(tf.random_normal([num_input,nodes_size]))
    biases[0] = tf.Variable(tf.random_normal([nodes_size]))
    layers[0] = tf.add(tf.matmul(Input,weights[0]), biases[0])
    
    # Hidden Layers
    for i in range(1, num_layers):
        weights[i] = tf.Variable(tf.random_normal([nodes_size, nodes_size]))
        biases[i] = tf.Variable(tf.random_normal([nodes_size]))
        layers[i] = tf.add(tf.matmul(layers[i-1], weights[i]), biases[i])

    # Output layer
    weights[num_layers] = tf.Variable(tf.random_normal([nodes_size, num_classes]))
    biases[num_layers] = tf.Variable(tf.random_normal([num_classes]))
    logits = tf.add(tf.matmul(layers[num_layers-1], weights[num_layers]), biases[num_layers])

    # get the predictions through a softmax
    prediction = tf.nn.softmax(logits)
    

    # It's Optimization Time! 
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Labels))
    
    #faster gradiend descent
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # We are going to use the MNIST Dataset to test the network
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("./data/", one_hot=True)


        sess.run(init)
    
        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={Input: batch_x, Labels: batch_y})
            
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={Input: batch_x,
                                                                     Labels: batch_y})
                print("Step " + str(step) + ", Loss= " + "{:.4f}".format(loss) + ", Accuracy= " + \
                      "{:.3f}".format(acc))
    
        print("Optimization Finished!")
    
        # Let's calculate the accuracy for MNIST test images
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={Input: mnist.test.images, Labels: mnist.test.labels}))

        print("Saving the model in:", save_path)
        saver.save(sess, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametized Perceptron Neural Network")
    parser.add_argument('--batch_size', type=int, default='128', help='How much data train on at a time')
    parser.add_argument('--steps', type=int, default='500', help='How many steps to take in the training')
    parser.add_argument('--learning_rate', type=float, default='0.1', help='How fast the learning rate is')
    parser.add_argument('--model_file', type=str, default='model.tfl', help='Path to save the model file')
    parser.add_argument('--data_path', type=str, default='/tmp/data', help='Path to save training data')
    parser.add_argument('--hidden_layer_num', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--nodes_size', type=int, default=256, help='Number of nodes in each layers')
    parser.add_argument('--class_number', type=int, default=10, help='Number of classes to train on')
    parser.add_argument('--input_size', type=int, default=784, help='Size of input dataset (i.e. MNIST example 28*28)')
    FLAGS = parser.parse_args()
    main()
    
    
