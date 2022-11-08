from PIL import Image
import matplotlib.pyplot as plt

def crop_doom(im, height, width):
    imgwidth, imgheight = im.size
    k = 1
    for i in range(5,imgheight,height+5):
        for j in range(5,imgwidth,width+5):
            box = (j, i, j+width, i+height)
            cropped_im = im.crop(box)
            plt.imshow(cropped_im)
            plt.show()
            path_file = "sprite_"+str(k)+".png"
            cropped_im.save(path_file)
            k+=1

im = Image.open("doom_textures.png")            
crop_doom(im,64,64)