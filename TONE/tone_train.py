import tensorflow as tf
import sys
from model import Model
from dataset import Dataset
from network import *
from datetime import datetime
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Dataset path
    train_color_list = '../idx/train_color_2tream_1.txt'
    train_thermal_list = '../idx/train_thermal_2tream_1.txt'
    checkpoint_path = 'log/'
    
    # Learning params
    learning_rate = 0.001
    training_iters = 1501
    batch_size = 64
    display_step = 5
    test_step = 500 # 0.5 epoch
    margin = 0.5
    
    # Network params
    n_classes = 206
    keep_rate = 0.5

    # Graph input
    x1 = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    x2 = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y0 = tf.placeholder(tf.float32, [batch_size*2, n_classes])
    y = tf.placeholder(tf.float32, [batch_size, 1])

    keep_var = tf.placeholder(tf.float32)

    # Model
    feat1 = Model().alexnet_visible(x1, keep_var)    
    feat2 = Model().alexnet_thermal(x2, keep_var)
    feat, pred0 = Model().share_modal(feat1, feat2, keep_var)
    
    # norm_feat = tf.nn.l2_normalize(feat, 1, epsilon=1e-12)
    feature1, feature2 = tf.split(0, 2, feat)
    feature1 = tf.nn.l2_normalize(feature1, 0, epsilon=1e-12)
    feature2 = tf.nn.l2_normalize(feature2, 0, epsilon=1e-12)
    dist = tf.reduce_sum(tf.square(feature1 - feature2), 1)
    d_sqrt = tf.sqrt(dist)
    loss0 = (1- y) * tf.square(tf.maximum(0., margin - d_sqrt)) + y * dist
    loss0 = 0.5 * tf.reduce_mean(loss0)
    
    # Loss and optimizer
    identity_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred0, y0))
    total_loss = identity_loss + 0.2* loss0
    # total_loss = identity_loss
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss)

    # Evaluation
    correct_pred0 = tf.equal(tf.argmax(pred0, 1), tf.argmax(y0, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred0, tf.float32))

    # Init
    init = tf.initialize_all_variables()

    # Load dataset
    dataset = Dataset(train_color_list, train_thermal_list)
    saver = tf.train.Saver()
    
    # Launch the graph
    with tf.Session() as sess:
        print 'Init variable'
        
        sess.run(init)
        
        print 'Start training'
        
        step = 0
        while step < training_iters:
            batch_x1, batch_x2, batch_y0, batch_y = dataset.next_batch(batch_size, 'train')
            
            sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y0:batch_y0, y:batch_y, keep_var: keep_rate})
            
            # Display training status
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, y0:batch_y0, y:batch_y, keep_var: 1.0})
                batch_loss = sess.run(total_loss, feed_dict={x1: batch_x1, x2: batch_x2, y0:batch_y0,y:batch_y,  keep_var: 1.0})
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f},".format(datetime.now(), step, batch_loss, acc)
            # Save the model checkpoint periodically.
            if step % 900 == 0 or step == training_iters:
                checkpoint_name = os.path.join(checkpoint_path, 'tone_iter2_'+ str(step) +'.ckpt')
                save_path = saver.save(sess, checkpoint_name)
            step += 1

        print "Finish!"

if __name__ == '__main__':
    main()

