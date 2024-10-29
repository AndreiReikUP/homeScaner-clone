import cv2
from PIL import Image
import pytesseract
from matplotlib import pyplot as plt
from numpy import ones, uint8

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
cv2.imwrite("temp/imagem_invertida.jpg", imagem_invertida) #Salvando a imagem

# "Binarização"
# Para essa conversão precisamos primeiro fazer um gray scale da imagem
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


imagem_cinza = grayscale(img)
cv2.imwrite("temp/image_cinza.jpg", imagem_cinza) #Salvando a imagem

# Conversão para BRANCO ou PRETO, 0 ou 1, TRUE ou FALSE
thresh, imagem_preto_e_branco = cv2.threshold(imagem_cinza, 210, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp/imagem_PeB.jpg", imagem_preto_e_branco) #Salvando a imagem

# Remoção de ruído na imagem
def noise_removal(image):
    kernel = ones((1, 1), uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = ones((1, 1), uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


no_noise = noise_removal(imagem_preto_e_branco)
cv2.imwrite("temp/no_noise.jpg", no_noise) #Salvando a imagem

# "Dilatação" e "Erosão" de imagem
# Ajuste de grossura da fonte

#Afinando a fonte
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = ones((2,2), uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

#Engrossando a fonte
def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = ones((2,2), uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

imagem_dilatada = thick_font(no_noise)
imagem_erodida = thin_font(no_noise)
cv2.imwrite("temp/imagem_dilatada.jpg", imagem_dilatada) #Salvando a imagem
cv2.imwrite("temp/imagem_erodida.jpg", imagem_erodida) #Salvando a imagem

#Remoção de borda
def remove_border(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return crop

no_borders = remove_border(no_noise)
cv2.imwrite("temp/no_borders.jpg", no_borders) #Salvando a imagem

#Adição de borda
#Dimensões da borda
color = [255, 255, 255]
top, bottom, left, right = [150]*4
with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

cv2.imwrite("temp/with_border.jpg", with_border) #Salvando a imagem