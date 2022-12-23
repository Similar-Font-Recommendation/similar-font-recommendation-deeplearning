from PIL import Image, ImageDraw, ImageFont
import os

def make_image(message):

    font_path = 'C:/Users/SM/Desktop/비정형/'
    
    # font setting
    font_list = os.listdir(font_path)
    font_color = 'rgba(0, 0, 0, 0)'
    font_size = 60
    
    # image size
    width = len(message) * font_size + 80
    height = font_size + 35
    bg_color = 'rgba(255, 255, 255, 255)'
    x_margin = 5
    y_margin = 5
    
    # 렌더링 결과 파일 저장할 폴더 생성
    img_save_path = 'C:/Users/SM/Desktop/2022-2 졸업프로젝트/Text Rendering/' + message + '(비)/'
    os.mkdir(img_save_path)
    
    for font in font_list:
      image =Image.new('RGB', (width, height), color = bg_color)
      font_name = font.replace('.ttf', '').replace('.TTF', '')
      font = ImageFont.truetype(font_path + font, font_size)
      draw = ImageDraw.Draw(image)      
      draw.text((x_margin, y_margin), message, font=font, fill=font_color)
      
      # save file
      image.save(img_save_path + font_name + '.png')
      
# 실행
make_image("훈민정음") 