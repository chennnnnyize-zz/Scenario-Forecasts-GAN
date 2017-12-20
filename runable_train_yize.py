import os
import pandas as pd
import numpy as np
from runable_model_yize import *
from load import load_wind_data_new
import matplotlib.pyplot as plt
from numpy import shape
from util import *

n_epochs = 50
learning_rate = 0.0002
batch_size = 32
image_shape = [24, 24, 1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1
k = 3
import csv

trX, trY, teX, teY, forecastX = load_wind_data_new()
print("shape of training samples ", shape(trX))
print("Wind data loaded")

def construct(X):
    X_new1=np.copy(X[:, 288:576])
    X_new_high=[x*1.2 for x in X_new1]
    X_new_low=[x*0.8 for x in X_new1]
    x_samples_high=np.concatenate((X[:, 0:288], X_new_high), axis=1)
    x_samples_high=np.clip(x_samples_high, 0.05, 0.95)
    x_samples_low = np.concatenate((X[:, 0:288], X_new_low), axis=1)
    x_samples_low = np.clip(x_samples_low, 0.05, 0.9)
    return x_samples_high, x_samples_low

def construct2(X):
    X_new=X[:, 288:576]
    X_new_high=[x*2.5 for x in X_new]
    #X_new_high=np.ones([32,288])
    X_new_low=[x*0.4 for x in X_new]
    #X_new_low=np.zeros([32,288])
    X_new_high=np.clip(X_new_high, 0.16, 1)
    x_samples_high=np.concatenate((X[:, 0:288], X_new_high), axis=1)
    X_new_low = np.clip(X_new_low, 0, 0.6)
    x_samples_low = np.concatenate((X[:, 0:288], X_new_low), axis=1)
    return x_samples_high, x_samples_low

def construct_hard(X):
    x_samples_high=np.ones(shape(X), dtype=float)
    x_samples_low=np.zeros(shape(X),dtype=float)
    for i in range(len(X)):
        m=np.mean(X[i,0:288])
        x_samples_high[i,:]=4*m*x_samples_high[i,:]
        x_samples_low[i, :] = 0.2 * m * x_samples_high[i, :]
    x_samples_high = np.clip(x_samples_high, 0, 1)
    return x_samples_high, x_samples_low

def plot(samples, X_real):
    m = 0
    f, axarr = plt.subplots(4, 8)
    for i in range(4):
        for j in range(8):
            axarr[i, j].plot(samples[m], linewidth=3.0)
            axarr[i, j].plot(X_real[m], 'r')
            axarr[i, j].set_xlim([0, 576])
            axarr[i, j].set_ylim([0, 16])
            m += 1
    plt.title('Comparison of predicted(blue) and real (red)')
    plt.savefig('comparison.png', bbox_inches='tight')
    plt.show()
    return f



def plot_sample(samples):
    m = 0
    f, axarr = plt.subplots(4, 8)
    for i in range(4):
        for j in range(8):
            axarr[i, j].plot(samples[m])
            axarr[i, j].set_xlim([0, 576])
            axarr[i, j].set_ylim([0, 16])
            m += 1
    plt.title('Generated samples')
    plt.savefig('generated_samples.png', bbox_inches='tight')
    plt.show()
    return f


dcgan_model = DCGAN(
    batch_size=batch_size,
    image_shape=image_shape,
    dim_z=dim_z,
    # W1,W2,W3: the dimension for convolutional layers
    dim_W1=dim_W1,
    dim_W2=dim_W2,
    dim_W3=dim_W3,
)
print("DCGAN model loaded")

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen, \
complete_loss, high_tf, low_tf, mask_tf, log_loss, loss_former, loss_prepare = dcgan_model.build_model()

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]

train_op_discrim = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-d_cost_tf, var_list=discrim_vars))
train_op_gen = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(g_cost_tf, var_list=gen_vars))
Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=batch_size)

Z_np_sample = np.random.uniform(-1, 1, size=(batch_size, dim_z))
Y_np_sample = OneHot(np.random.randint(5, size=[batch_size]), n=5)
iterations = 0
P_real = []
P_fake = []
P_distri = []
discrim_loss = []

with tf.Session() as sess:
    # begin training
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    '''ckpt = tf.train.get_checkpoint_state('model.ckpt')
    print("CKPt", ckpt)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, 'model.ckpt')
      print(" [*] Success to read!")
    else: print("model load failed: here")'''
    #saver.restore(sess, 'model.ckpt.data-00000-of-00001')


    print("Number of batches in each epoch:", len(trY) / batch_size)
    for epoch in range(n_epochs):
        print("epoch" + str(epoch))
        index = np.arange(len(trY))
        np.random.shuffle(index)
        trX = trX[index]
        trY = trY[index]
        trY2 = OneHot(trY, n=5)
        for start, end in zip(
                range(0, len(trY), batch_size),
                range(batch_size, len(trY), batch_size)
        ):

            Xs = trX[start:end].reshape([-1, 24, 24, 1])
            Ys = trY2[start:end]
            # use uniform distribution data to generate adversarial samples
            Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

            # for each iteration, generate g and d respectively, k=2
            if np.mod(iterations, k) != 0:
                _, gen_loss_val = sess.run(
                    [train_op_gen, g_cost_tf],
                    feed_dict={
                        Z_tf: Zs,
                        Y_tf: Ys
                    })
            

            else:
                _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={
                        Z_tf: Zs,
                        Y_tf: Ys,
                        image_tf: Xs
                    })
                # gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen], feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})
            p_real_val, p_gen_val = sess.run([p_real, p_gen], feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
            P_real.append(p_real_val.mean())
            P_fake.append(p_gen_val.mean())
            #discrim_loss.append(discrim_loss_val)

            if np.mod(iterations, 5000) == 0:
                print("iterations ", iterations)
                gen_loss_val, discrim_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, d_cost_tf, p_real, p_gen],
                                                                                 feed_dict={Z_tf: Zs, image_tf: Xs,
                                                                                            Y_tf: Ys})
                print("Average P(real)=", p_real_val.mean())
                print("Average P(gen)=", p_gen_val.mean())
                print("discrim loss:", discrim_loss_val)
                print("gen loss:", gen_loss_val)

                Z_np_sample = np.random.uniform(-1, 1, size=(batch_size, dim_z))
                generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample: Z_np_sample,
                        Y_tf_sample: Y_np_sample
                    })
                generated_samples=generated_samples.reshape([-1,576])
                generated_samples = generated_samples * 16
                #save_visualization(generated_samples, (8, 8), save_path='./test/sample_' + str(iterations) + '.jpg')
                csvfile=file('%s.csv'%iterations, 'wb')
                writer=csv.writer(csvfile)
                writer.writerows(generated_samples)
            iterations = iterations + 1
    '''plt.plot(P_real)
    plt.plot(P_fake)
    plt.show()'''

    '''save_path = saver.save(sess,
               'model.ckpt'
               )
    print("Model saved in path: %s"%save_path)'''


    print("Start to generate scenarios")
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.001
    iterations = 0

    completed_samples = []
    mask = np.ones([batch_size, 24, 24, 1])
    mask[:, 12:24, :, :] = 0.0

    for start, end in zip(
            range(0, len(forecastX), batch_size),
            range(batch_size, len(forecastX),  batch_size)
    ):
        print("ready to generate scenarios in iteration %s", iterations)
        forecast_samples = forecastX[start:end]
        Xs=teX[start:end]
        X_feed_high, X_feed_low = construct(forecast_samples)
        X_feed_high2, X_feed_low2 = construct2(forecast_samples)
        Ys = teY[start:end]
        Ys = OneHot(Ys, n=5)

        csvfile = file('orig_iter%s.csv' % iterations, 'wb')
        writer = csv.writer(csvfile)
        orig_samples = Xs * 16
        writer.writerows(orig_samples)
        csvfile = file('forecast_iter%s.csv' % iterations, 'wb')
        writer = csv.writer(csvfile)
        orig_samples = forecast_samples * 16
        writer.writerows(orig_samples)
        csvfile = file('forhigh_iter%s.csv' % iterations, 'wb')
        writer = csv.writer(csvfile)
        orig_samples = X_feed_high2 * 16
        writer.writerows(orig_samples)
        csvfile = file('forlow_iter%s.csv' % iterations, 'wb')
        writer = csv.writer(csvfile)
        orig_samples = X_feed_low2 * 16
        writer.writerows(orig_samples)


        '''plt.plot(X_feed_high[0],'b')
        plt.plot(X_feed_low[0],'r')
        plt.plot(Xs[0],'g')
        plt.show()'''

        '''fig = plt.figure()
        fig.set_figheight(40)
        fig.set_figwidth(80)
        for m in range(32):
            ax = fig.add_subplot(4, 8, m + 1)
            ax.plot(orig_samples[m], color='b')
            ax.plot(X_feed_high2[m]*16, color='g')
            ax.plot(X_feed_low2[m]*16, color='y')'''

        Xs_shaped = Xs.reshape([-1, 24, 24, 1])
        samples=[]


        for batch in range(120):
            print("Batch:", batch)
            zhats = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)
            image_pre=np.zeros([batch_size, 576])
            for i in range(batch_size):
                for j in range(288, 576):
                    image_pre[i][j] = np.random.uniform(X_feed_low[i,j], X_feed_high[i,j])


            image_pre=image_pre.reshape([-1, 24, 24, 1])
            m = 0
            v = 0
            for i in xrange(1200):
                fd = {
                    Z_tf: zhats,
                    image_tf: image_pre,
                    Y_tf: Ys,
                    mask_tf: mask,
                }

                g, = sess.run([loss_prepare], feed_dict=fd)

                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = beta1 * m_prev + (1 - beta1) * g[0]
                v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))

                zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                zhats = np.clip(zhats, -1, 1)

                '''if np.mod(i, 500) == 0:
                    print("Gradient iteration:", i)'''


            image_pre=image_pre.reshape([-1,576])

            '''plt.plot(generated_samples[0])
            plt.plot(image_pre[0]*16)
            plt.show()'''

            m = 0
            v = 0

            for i in xrange(1000):
                fd = {
                    Z_tf: zhats,
                    image_tf: Xs_shaped,
                    Y_tf: Ys,
                    high_tf: X_feed_high2.reshape([-1,24,24,1]),
                    low_tf: X_feed_low2.reshape([-1,24,24,1]),
                    mask_tf: mask,
                }

                g, log_loss_value, sample_loss_value = sess.run([complete_loss, log_loss, loss_former], feed_dict=fd)

                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = beta1 * m_prev + (1 - beta1) * g[0]
                v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))

                zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                zhats = np.clip(zhats, -1, 1)

                #if np.mod(i, 200) == 0:
                    #print("Gradient iteration:", i)
                    #print("Log loss", log_loss_value[0])
                    #print("Sample loss", sample_loss_value)

                '''generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample: zhats,
                        Y_tf_sample: Ys
                    })

                generated_samples = generated_samples.reshape(32, 576)
                generated_samples = generated_samples * 16
                plt.plot(generated_samples[0],'r')
                plt.plot(image_pre[0]*16, 'k')
                #plt.plot(generated_samples[1],'r')
                plt.plot(X_feed_high2[0]*16,'y')
                plt.plot(X_feed_low2[0]*16,'y')
                plt.plot(orig_samples[0],'b')
                #plt.plot(orig_samples[1],'b')
                plt.plot(X_feed_low[0]*16,'g')
                #plt.plot(X_feed_low[1] * 16, 'g')
                plt.plot(X_feed_high[0] * 16, 'g')
                #plt.plot(X_feed_high[1] * 16, 'g')
                plt.show()'''

            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample: zhats,
                    Y_tf_sample: Ys
                })

            generated_samples = generated_samples.reshape(32, 576)
            samples.append(generated_samples)

            '''plt.plot(generated_samples[0],color='r')
            plt.plot(X_feed_low[0]*16, color='g')
            plt.plot(X_feed_high[0]*16, color='y')
            plt.plot(orig_samples[0], color='b')
            plt.show()'''
            '''csvfile = file('generated_iter%sgroup%s.csv' % (iterations, batch), 'wb')
            writer = csv.writer(csvfile) 
            writer.writerows(generated_samples)'''

            '''for m in range(32):
                ax2 = fig.add_subplot(4, 8, m + 1)
                ax2.plot(generated_samples[m], color='r')


        
        fig.savefig('generated_iter%s.png'% (iterations))
        plt.close(fig)
        iterations += 1'''
        samples=np.array(samples, dtype=float)
        '''print(shape(samples))
        samples=samples.reshape([-1,12])
        samples=np.mean(samples,axis=1)
        samples=samples.reshape([-1,48])'''
        print(shape(samples))
        samples=samples*16
        csvfile = file('generated_iter%s.csv' % iterations, 'wb')
        writer = csv.writer(csvfile)
        writer.writerows(samples.reshape([-1,576]))
        iterations+=1

