from PIL import Image
import os

def trim(img):
  pixels = img.load()
  # print (f"original: {img.size[0]} x {img.size[1]}")
  
  xlist = []
  ylist = []
  for y in range(0, img.size[1]):
    for x in range(0, img.size[0]):
      if pixels[x, y] != (255, 255, 255):
        xlist.append(x)
        ylist.append(y)
  left = min(xlist)
  right = max(xlist)
  top = min(ylist)
  bottom = max(ylist)
  
  img = img.crop((left, top, right, bottom))
  # print (f"cropped: {img.size[0]} x {img.size[1]}")
  return img
  
imgs_path = "img_load_path"
img_list = os.listdir(imgs_path)
for img_name in img_list:
  img = Image.open(imgs_path + img_name)
  trimed_img = trim(img)
  trimed_img.save("img_save_path" + img_name)
# trimed_img.show()