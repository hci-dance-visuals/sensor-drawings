# -*- coding: utf-8 -*-
"""
    DISCLAIMER:
    This code has been written in the optic
    of my 'quick-n-dirty' Deep Learning series
    on Medium (@juliendespois) to show the
    concepts. Please do not judge me by the
    quality of the code.
    ¯\_(ツ)_/¯
    """
from tornado import websocket, web, ioloop
import thread

import sys
import time, math
import cv2

from config import latent_dim, modelsPath, imageSize
from model import getModels
from visuals import visualizeDataset, visualizeReconstructedImages, computeTSNEProjectionOfLatentSpace,computeTSNEProjectionOfPixelSpace, visualizeInterpolation, visualizeArithmetics, getInterpolatedFrames, layer_images
from datasetTools import loadDataset, loadDatasetLabelled, loadDatasetAugmented
from config import latent_dim, imageSize
import numpy as np
import tensorflow as tf
from keras.optimizers import RMSprop
from random import randint

from numpy_ringbuffer import RingBuffer
from pyImageStreamer import PyImageStreamer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Handy parameters
nbEpoch = 50
batchSize = 8
imSize = imageSize[0]
modelName = "autoencoder_modi_aug.h5"

#Run ID for tensorboard, timestamp is for ordering
runID = "{} - Autoencoder - MNIST".format(1./time.time())

osc_server = None
osc_listening_port = 12001
ws_send_port = 8882
scrub_position = 0
likeliest = 0
nsteps = 50
window_size = 15
scrub_input = RingBuffer(capacity=window_size, dtype=int)
delay = 0
smudge_enabled = True
smudge_amt = 0.5
id = 6
gesture_selection = [3,6,7,9]
cl = []

# Returns the string of remaining training time
def getETA(batchTime, nbBatch, batchIndex, nbEpoch, epoch):
    seconds = int(batchTime*(nbBatch-batchIndex-1) + batchTime*nbBatch*(nbEpoch-epoch-1))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d"%(h,m,s)

# Trains the Autoencoder, resume training with startEpoch > 0
def trainModel(startEpoch=0):
    # Create models
    print("Creating Autoencoder...")
    autoencoder, _, _ = getModels()
    autoencoder.compile(optimizer=RMSprop(lr=0.00025), loss="mse")

    # From which we start
    if startEpoch > 0:
        # Load Autoencoder weights
        print("Loading weights...")
        autoencoder.load_weights(modelsPath+modelName)

    print("Loading dataset...")
    X_train, X_test, Y_train = loadDatasetAugmented(imSize)

    # Compute number of batches
    nbBatch = int(X_train.shape[0]/batchSize)

    # Train the Autoencoder on dataset
    print ("Training Autoencoder for {} epochs with {} batches per epoch and {} samples per batch.".format(nbEpoch,nbBatch,batchSize))
    print ("Run id: {}".format(runID))

    # Debug utils writer
    writer = tf.summary.FileWriter("/tmp/logs/"+runID)
    batchTimes = [0. for i in range(5)]

    # For each epoch
    for epoch in range(startEpoch,nbEpoch):
        # For each batch
        for batchIndex in range(nbBatch):
            batchStartTime = time.time()
            # Get batch
            X = X_train[batchIndex*batchSize:(batchIndex+1)*batchSize]

            # Train on batch
            autoencoderLoss = autoencoder.train_on_batch(X, X)
            trainingSummary = tf.Summary.Value(tag="Loss", simple_value=float(autoencoderLoss))

            # Compute ETA
            batchTime = time.time() - batchStartTime
            batchTimes = batchTimes[1:] + [batchTime]
            eta = getETA(sum(batchTimes)/len(batchTimes), nbBatch, batchIndex, nbEpoch, epoch)

            # Save reconstructions on train/test samples
            if batchIndex%2 == 0:
                visualizeReconstructedImages(X_train[:16],X_test[:16],autoencoder, save=True, label="{}_{}".format(epoch,batchIndex))

            # Validation & Tensorboard Debug
            if batchIndex%20 == 0:
                validationLoss = autoencoder.evaluate(X_test[:512], X_test[:512], batch_size=256, verbose=0)
                if len(validationLoss) != 0:
                    validationSummary = tf.Summary.Value(tag="Validation Loss", simple_value=float(validationLoss[0]))
                    summary = tf.Summary(value=[trainingSummary,validationSummary])
                    print ("Epoch {}/{} - Batch {}/{} - Loss: {:.3f}/{:.3f} - ETA:".format(epoch+1,nbEpoch,batchIndex+1,nbBatch,autoencoderLoss,validationLoss), eta)
#                print ("Epoch {}/{} - Batch {}/{} - ETA:".format(epoch+1,nbEpoch,batchIndex+1,nbBatch,autoencoderLoss), eta)
                    writer.add_summary(summary, epoch*nbBatch + batchIndex)
            else:
                print ("Epoch {}/{} - Batch {}/{} - Loss: {:.3f} - ETA:".format(epoch+1,nbEpoch,batchIndex+1,nbBatch,autoencoderLoss), eta)
                summary = tf.Summary(value=[trainingSummary,])
                writer.add_summary(summary, epoch*nbBatch + batchIndex)

        #Save model every epoch
        print("Saving autoencoder...")
        autoencoder.save_weights(modelsPath+modelName, overwrite=True)

# Generates images and plots
def testModel(stream_output=False):
    # Create models
    print("Creating Autoencoder, Encoder and Generator...")
    autoencoder, encoder, decoder = getModels()

    # Load Autoencoder weights
    print("Loading weights...")
    autoencoder.load_weights(modelsPath+modelName)

    # Load dataset to test
    print("Loading dataset...")
    if "aug" in modelName:
        X_train, X_test, Y_train = loadDatasetAugmented(imSize)
    else:
        X_train, X_test, Y_train = loadDatasetLabelled(imSize)
    name_list = np.unique(Y_train)
    print(X_train.shape)
#    print(name_list)
    # Visualization functions
    #visualizeReconstructedImages(X_train[:180],X_test[:20], autoencoder)
    #     computeTSNEProjectionOfPixelSpace(X_test[:1000], display=True)
    #     computeTSNEProjectionOfLatentSpace(X_train[:1000], encoder, display=True)
    lkst = likeliest
    scrub = 0
    seed = (id*20) + gesture_selection[likeliest]
#    seed = (id*20) + (likeliest*4)
    scrub_destination = seed + 4
    name = Y_train[seed]
    ri = getInterpolatedFrames(X_train[randint(0,X_train.shape[0])],
    X_train[randint(0,X_train.shape[0])], encoder, decoder, save=False, nbSteps=nsteps)
#    ri = getInterpolatedFrames(X_train[seed], X_train[scrub_destination], encoder, decoder, save=False, nbSteps=nsteps)
#    return
    while likeliest == lkst:
        osc_server.recv(10)
        if (smudge_enabled):
            if(scrub_input.shape[0] > 0):
                frame_buffer = [ri[sc] for sc in scrub_input]
                img_out = layer_images(frame_buffer, smudge_amt)
            else:
                continue
        else:
            img_out = ri[int(math.floor(scrub_position*(nsteps-1)))]
        if (stream_output):
            png_bytes = pyImageStreamer.get_jpeg_image_bytes(img_out)
            if len(cl)>0: cl[-1].write_message(png_bytes, binary=True)
        else:
            cv2.imshow('Moving_Digits',img_out)
            cv2.waitKey(2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
    testModel(stream_output)

def update_scrub(path, args):
    global scrub_position
    global scrub_input
    scrub_position = args[0]
#    scrub_position = args[likeliest]
    scrub_input.appendleft(int(math.floor(scrub_position*49)))

def update_likeliest(path, args):
    global likeliest
    likeliest = args[0]
    print("likeliest: %s" % likeliest)

def fallback(path, args, types, src):
    print ("got unknown message '%s' from '%s'") % (path, src.get_url())
    for a, t in zip(args, types):
        print ("argument of type '%s': %s") % (t, a)

def init_osc_server():
    print("importing OSC libs")
    import liblo
    # config for listening
    listening_port = osc_listening_port
    # create server, listening on given port
    try:
        server = liblo.Server(listening_port)
    except liblo.ServerError, err:
        print str(err)
        sys.exit()
    server.add_method("/modi/scrub", 'ffff', update_scrub)
    server.add_method("/modi/likeliest", 'i', update_likeliest)
    server.add_method(None, None, fallback)
    return server
#    thread.start_new_thread(osc_input_handler, (server,))

pyImageStreamer = PyImageStreamer(50, 1, ws_send_port, 1.0)
class SocketHandler(websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        if self not in cl:
            cl.append(self)
        pyImageStreamer.request_start()
        print("CONNECTED")

    def on_message(self, message):
        self.write_message(u"You said: " + message)

    def on_close(self):
        if self in cl:
            cl.remove(self)
        print("DISCONNECTED")
        pyImageStreamer.request_stop()

def signal_handler(signal, frame):
    print('TERMINATED')
    sys.exit(0)

def data_handler(imgStreamer):
    while 1:
        png_bytes = imgStreamer.get_jpeg_image_bytes()
        if len(cl)>0: cl[-1].write_message(png_bytes, binary=True)

def export_as_tf():
    print "exporting"
    from keras import backend as K
    from keras.models import load_model
    autoencoder, encoder, decoder = getModels()
    autoencoder.load_weights(modelsPath+modelName)
    saver = tf.train.Saver()
    sess = K.get_session()
    save_path = saver.save(sess, "Models/tf_modi")


app = web.Application([(r'/', SocketHandler)])

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) == 2 else None
    if arg is None:
        print ("Need argument")
    elif arg == "train":
        trainModel(startEpoch=0)
    elif arg == "test":
        osc_server = init_osc_server()
        testModel(stream_output=False)
        ioloop.IOLoop.current().start()
    elif arg == "stream":
        osc_server = init_osc_server()
        print("Sending bytes to: http://localhost:" + str(pyImageStreamer.port) + "/")
        app.listen(pyImageStreamer.port)
        thread.start_new_thread(testModel, (True,))
        ioloop.IOLoop.current().start()
    elif arg == "export":
        export_as_tf()
    else:
        print ("Wrong argument")

#    for i in range(50):
#        # seed = randint(0,X_train.shape[0]-1)
#        scrub = randint(0, 50)
#        id = 5
#        seed = (id*20) + 0
#        name = Y_train[seed]
#        print(name, seed)
#        scrub_destination = seed + 1
#        visualizeInterpolation(X_train[seed], X_train[scrub_destination], encoder, decoder, save=False, nbSteps=50, count=c)
