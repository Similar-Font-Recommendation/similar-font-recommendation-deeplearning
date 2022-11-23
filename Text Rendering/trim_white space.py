from PIL import Image
def trim(img):
  pixels = img.load()

  print (f"original: {img.size[0]} x {img.size[1]}")
  xlist = []
  ylist = []
  for y in range(0, img.size[1]):
    for x in range(0, img.size[0]):
      if pixels[x, y] != (255, 255, 255, 255):
        xlist.append(x)
        ylist.append(y)
  left = min(xlist)
  right = max(xlist)
  top = min(ylist)
  bottom = max(ylist)
  
  img = img.crop((left, top, right, bottom))
  print (f"cropped: {img.size[0]} x {img.size[1]}")
  return img
  
img = Image.open("test_img.PNG")
trimed_img = trim(img)
trimed_img.show()