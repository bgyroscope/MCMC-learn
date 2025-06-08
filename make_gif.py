"""
    make a quick gif 
    probably from SamplerTester.plot(save_frame=True)
"""
import os.path
import imageio

fold = 'temp'


files = [file for file in os.listdir(fold) if file[-3:] == 'png'] 
files.sort()

images = [] 
for file in files:
    images.append(imageio.imread(os.path.join(fold,file)))
imageio.mimsave(os.path.join(fold,'temp.gif'),images, fps=10)

print(files)