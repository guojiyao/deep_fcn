import urllib
import os
testfile = urllib.URLopener()
#testfile.retrieve("https://s3.amazonaws.com/crowdai-ml-challenge/masks/3band_013022223130_Public_img1.tif", "masks_img1.tif")
#testfile.retrieve("https://s3.amazonaws.com/crowdai-ml-challenge/tiles/3band_013022223130_Public_img1.tif", "tiles_img1.tif")

text_file = open("masks_urls.txt", "r")
lines = text_file.readlines()

text_file_2 = open("tiles_urls.txt", "r")
lines_2 = text_file_2.readlines()


#os.system('wget ' + "https://s3.amazonaws.com/crowdai-ml-challenge/masks/3band_013022223130_Public_img1.tif")
for i in range (7000):
	os.system('wget ' + lines[i][:-1]+" -O data/masks/masks_img"+str(i+1)+".tif")

for i in range (7000):
	os.system('wget ' + lines_2[i][:-1]+" -O data/tiles/tiles_img"+str(i+1)+".tif")

#for i in range (5000):
#	testfile.retrieve(lines_2[i][:-1],"tiles/tiles_img"+str(i+1)+".tif")
