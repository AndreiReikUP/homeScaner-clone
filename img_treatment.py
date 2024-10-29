import cv2
from PIL import Image
import pytesseract
from matplotlib import pyplot as plt

image_file = "assets/page_01.jpg"
img = cv2.imread(image_file)

# Abrindo a imagem com openCV
# Função para fazer o display correto da imagem:
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    figsize = width/float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])

    ax.axis('off')

    ax.imshow(im_data, cmap='gray')
    
    plt.show()

# Imagens invertidas 
# (Atualmente não é tão eficiente, porém para versão 3.X abaixo do tesseract ainda funcionava)
imagem_invertida = cv2.bitwise_not(img)
cv2.imwrite("temp/imagem_invertida.jpg", imagem_invertida)

# "Binarização"
# Para essa conversão precisamos primeiro fazer um gray scale da imagem
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imagem_cinza = grayscale(img)
cv2.imwrite("temp/image_cinza.jpg", imagem_cinza)

# Conversão para BRANCO ou PRETO, 0 ou 1, TRUE ou FALSE
thresh, imagem_preto_e_branco = cv2.threshold(imagem_cinza, 210, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp/imagem_PeB.jpg", imagem_preto_e_branco)

# Remoção de ruído na imagem
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

no_noise = noise_removal(imagem_preto_e_branco)
cv2.imwrite("temp/no_noise.jpg", no_noise)

# "Dilatação" e "Erosão" de imagem
# Ajuste de grossura da fonte
