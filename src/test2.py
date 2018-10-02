from os.path import isfile
import imageio
from pprint import pprint
import matplotlib.pyplot as plt
plt.ion()

videofile = 'ball2.mp4'
#if not isfile(videofile):
    #print('Downloading the video')
    #wget.download('http://staff.science.uva.nl/R.vandenBoomgaard/IPCV20162017/_static/ball2.mp4')
imageio.plugins.ffmpeg.download()
reader = imageio.get_reader(videofile)

pprint(reader.get_meta_data())

img = None
txt = None
prev_im = None
for i in range(40,400,4):
    im = reader.get_data(i)
    if img is None:
        img = plt.imshow(im)
    else:
        img.set_data(im)
   
    plt.pause(0.01)
    plt.draw()
    if prev_im is not None:
        print((im==prev_im).all())
    
    prev_im = im.copy()
