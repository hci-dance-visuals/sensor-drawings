import os
import sys
import h5py
import cv2
import math
import random, string

import numpy as np
from scipy.stats import norm
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from imutils import paths

from model import getModels
from config import latent_dim, imageSize, visualsPath
from datasetTools import loadDataset

imageH = imageSize[0]
imageW = imageSize[1]

def layer_images(frameBuffer, exp):
    imageBuffer = frameBuffer
    prevImg = 0
    combined_heatmaps = 0
    #use first frame as seed
    firstFrame = 0
    img = imageBuffer[firstFrame]
    if img is not None:
        prevImg = np.copy(img) #copy seed and fill with black pixels
        prevImg[prevImg > 0] = 0 #np.zeros wasn't happening
        combined_heatmaps = np.copy(prevImg)
    for file in imageBuffer:
        img = file
        #print(len(prevImg.shape))
        if img is not None:
            #layer the following frame
            combined_heatmaps = cv2.addWeighted(prevImg, exp, img, 1, 0)
            prevImg = combined_heatmaps #copy the result and repeat
    #imgString = cv2.imencode(".png",combined_heatmaps)
    return combined_heatmaps

def create_gif(inputPath, outputPath, delay, finalDelay, loop):
    # grab all image paths in the input directory
    imagePaths = sorted(list(paths.list_images(inputPath)))
    # remove the last image path in the list
    lastPath = imagePaths[-1]
    imagePaths = imagePaths[:-1]
    # construct the image magick 'convert' command that will be used
    # generate our output GIF, giving a larger delay to the final
    # frame (if so desired)
    cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
        delay, " ".join(imagePaths), finalDelay, lastPath, loop,
        outputPath)
    # cycle gif
    cmd += "&& convert %s -coalesce -duplicate 1,-2-1 -quiet -layers OptimizePlus  -loop 0 %s" % (outputPath, outputPath)
    # nomalize framerate
    cmd += "&& convert -delay 1x25 %s %s" % (outputPath, outputPath)
    os.system(cmd)

# Show every image, good for picking interplation candidates
def visualizeDataset(X):
    for i,image in enumerate(X):
        cv2.imshow(str(i),image)
        cv2.waitKey()
        cv2.destroyAllWindows()

# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([imageSize[0],imageSize[1]])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(X, encoder, display=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder.predict(X)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.5)
        plt.show()
    else:
        return X_tsne

# Show dataset images with T-sne projection of pixel space
def computeTSNEProjectionOfPixelSpace(X, display=True):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.reshape([-1,imageSize[0]*imageSize[1]*1]))

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.6)
        plt.show()
    else:
        return X_tsne

# Reconstructions for samples in dataset
def getReconstructedImages(X, autoencoder):
    nbSamples = X.shape[0]
    nbSquares = int(math.sqrt(nbSamples))
    nbSquaresHeight = 2*nbSquares
    nbSquaresWidth = nbSquaresHeight
    resultImage = np.zeros((nbSquaresHeight*imageH, nbSquaresWidth*imageH/2,X.shape[-1]))
    reconstructedX = autoencoder.predict(X)
    return reconstructedX
    for i in range(nbSamples):
        original = X[i]
        reconstruction = reconstructedX[i]
        rowIndex = i%nbSquaresWidth
        columnIndex = (i-rowIndex)/nbSquaresHeight
        resultImage[rowIndex*imageSize[0]:(rowIndex+1)*imageSize[0],columnIndex*2*imageSize[0]:(columnIndex+1)*2*imageSize[0],:] = np.hstack([original,reconstruction])

    return resultImage

# Reconstructions for samples in dataset
def visualizeReconstructedImages(X_train, X_test, autoencoder, save=False, label=False):
	trainReconstruction = getReconstructedImages(X_train,autoencoder)
	testReconstruction = getReconstructedImages(X_test,autoencoder)

	if not save:
		print("Generating 10 image reconstructions...")

	#     result = np.hstack([trainReconstruction,np.zeros([trainReconstruction.shape[0],5,trainReconstruction.shape[-1]]),testReconstruction])
	#     result = (result*255.).astype(np.uint8)
	results = trainReconstruction
	for i, result in enumerate(results):
		if save:
			cv2.imwrite(visualsPath + str(i)+".png",result)
		else:
			result = cv2.resize(result, dsize=(108, 216), interpolation=cv2.INTER_CUBIC)
			cv2.imshow("Reconstructed images (train - test)", result)
			cv2.waitKey(500)
			cv2.destroyAllWindows()

	# Computes A, B, C, A+B, A+B-C in latent space
def visualizeArithmetics(a, b, c, encoder, decoder):
    print("Computing arithmetics...")
    # Create micro batch
    X = np.array([a,b,c])

    # Compute latent space projection
    latentA, latentB, latentC = encoder.predict(X)

    add = latentA+latentB
    addSub = latentA+latentB-latentC

    # Create micro batch
    X = np.array([latentA,lagtentB,latentC,add,addSub])

    # Compute reconstruction
    reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub = decoder.predict(X)

    cv2.imshow("Arithmetics in latent space",np.hstack([reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub]))
    cv2.waitKey()

# Shows linear inteprolation in image space vs latent space
def visualizeInterpolation(start, end, encoder, decoder, save=False, nbSteps=5, count=0):
    print("Generating interpolations...")

    # Create micro batch
    X = np.array([start,end])

    # Compute latent space projection
    latentX = encoder.predict(X)
    latentStart, latentEnd = latentX

    # Get original image for comparison
    startImage, endImage = X

    vectors = []
    normalImages = []
    #Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    for alpha in alphaValues:
    	# Latent space interpolation
    	vector = latentStart*(1-alpha) + latentEnd*alpha
    	vectors.append(vector)
    	# Image space interpolation
    	blendImage = cv2.addWeighted(startImage,1-alpha,endImage,alpha,0)
    	normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = np.array(vectors)
    reconstructions = decoder.predict(vectors)

    # Put final image together
    resultLatent = None
    resultImage = None

    if save:
    	hashName = ''.join(random.choice(string.lowercase) for i in range(3))

    for i in range(len(reconstructions)):
    	interpolatedImage = normalImages[i]*255
    	interpolatedImage = cv2.resize(interpolatedImage,(256,502))
    	interpolatedImage = interpolatedImage.astype(np.uint8)
    	resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage,interpolatedImage])

    	reconstructedImage = reconstructions[i]*255.
    	reconstructedImage = reconstructedImage.reshape([112,112])
    	reconstructedImage = cv2.resize(reconstructedImage,(256,502))
    	reconstructedImage = reconstructedImage.astype(np.uint8)
    	resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent,reconstructedImage])

    	if save:
    		cv2.imwrite(visualsPath+"{}_{}.png".format(hashName,i),resultLatent)

    	result = np.vstack([resultLatent])

        id = str("%03d" % i)
    	# cv2.imwrite("./output/tests/result_%d/step_%s.png" % (count, id),reconstructedImage)
        cv2.imshow('Moving_Digits',reconstructedImage)
        cv2.waitKey(50)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite("./output/result_%d.png"%count,resultLatent)
    # create_gif("./output/tests/result_%d/"%count, "./output/tests/result_%d/animation.gif"%count, 0, 0, 0)

def getInterpolatedFrames(start, end, encoder, decoder, save=False, nbSteps=5, pID = 0, scrub=0):
    print("Generating interpolations...")

    # Create micro batch
    X = np.array([start,end])

    # Compute latent space projection
    latentX = encoder.predict(X)
    latentStart, latentEnd = latentX
    print ("latent start: ", latentStart.shape[0])
    print ("latent  end: ", latentEnd.shape[0])
    # Get original image for comparison
    startImage, endImage = X

    vectors = []
    normalImages = []
    #Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart*(1-alpha) + latentEnd*alpha
        vectors.append(vector)
        # Image space interpolation
        blendImage = cv2.addWeighted(startImage,1-alpha,endImage,alpha,0)
        normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = np.array(vectors)
    reconstructions = decoder.predict(vectors)
    reconstructedImages = []

    for i in range(len(reconstructions)):
        reconstructedImage = reconstructions[i]*255.
        reconstructedImage = reconstructedImage.reshape([112,112])
        reconstructedImage = cv2.resize(reconstructedImage,(256,502))
#        cv2.imwrite("Visuals/"+str(i)+'.png', reconstructedImage)
        reconstructedImage = reconstructedImage.astype(np.uint8)
        reconstructedImages.append(reconstructedImage)
    print("Finished interpolating sequence")
    return reconstructedImages

    def getInterpolatedFrames(start, end, encoder, decoder, save=False, nbSteps=5, pID = 0, scrub=0):
        return 0


#def scrub_images(reconstructedImages, scrub):
#    cv2.imshow('Moving_Digits',reconstructedImages[scrub])
#    cv2.waitKey(50)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        return

if __name__ == "__main__":
    # Load dataset to test
    print("Loading dataset...")
    X_train, X_test = loadDataset()
    visualizeDataset(X_test[:100])
