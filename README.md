# sensor-drawings

This project is used to morph between the bodily drawings collectied during the Moving Digits workshops.
![sample](https://user-images.githubusercontent.com/9369774/65178744-51238800-da51-11e9-9fc3-917ba4970709.png)

The model infractructure and code base is adapted from the following project by [Julien Despois](https://github.com/despoisj):
[https://hackernoon.com/latent-space-visualization-deep-learning-bits-2-bd09a46920df](https://hackernoon.com/latent-space-visualization-deep-learning-bits-2-bd09a46920df)

### Dependancies
- Python 2.7
- Keras (Tensorflow Backend)
- python-opencv
- numpy
- numpy_ringbuffer
- liblo (for OSC interaction) 

### Instructions to run

To execute the visualisations with a pre-trianed model run

    python main.py test
    
Alternitavley, use the **stream** argument to forward the output to WebSockets

To train a new model, set the appicable file directories in `config.py` and run

    python main.py train
    
### OSC Interaction

The following OSC messages sent to port **12001** can be used to control the output in real-time

`/modi/scrub` followed by a list of floats will update the interpolation position

`/modi/likilest` followed by an integer will change the seed image
    
    
