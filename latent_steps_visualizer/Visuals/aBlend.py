from PIL import Image

# Function to change the image size
def changeImageSize(maxWidth,
                    maxHeight,
                    image):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]
    
    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])
    
    newImage    = image.resize((newWidth, newHeight))
    return newImage

# Take two images for blending them together
image1 = Image.open("0.png")
image2 = Image.open("9.png")

# Make sure images got an alpha channel
image5 = image1.convert("RGBA")
image6 = image2.convert("RGBA")

# alpha-blend the images with varying values of alpha
for a in range(10):
    alphaBlended1 = Image.blend(image5, image6, alpha=a/10.)
    alphaBlended1.save('blended/'+ str(a) + '.png')
# Display the alpha-blended images
#alphaBlended1.show()
#alphaBlended2.show()

