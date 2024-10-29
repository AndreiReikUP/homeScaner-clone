import cv2
import numpy as np
import matplotlib.pyplot as plt

#Imagem de template 
refFilename = 'assets/FotoRefExa.png'
print('Lendo imagem de referência: ', refFilename)
im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

#Imagem à ser corrigida
imFilename = 'assets/FotoCorrExa.png'
print('Lendo imagem à ser corrigida: ', imFilename)
im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# #Mostrando as imagens
plt.figure(figsize=[20,10])
plt.subplot(121); plt.axis('off'); plt.imshow(im1); plt.title('Foto Original')
plt.subplot(122); plt.axis('off'); plt.imshow(im2); plt.title('Foto Scanneada')

#Encontrando pontos iguais em ambas as imagens
#Converter para grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

#Detectar elementos com o algoritmo ORB e computar os descritivos
MAX_NUM_FEATURES = 500
orb = cv2.ORB_create(MAX_NUM_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

#Mostrando imagens pós detecção
im1_display = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
im2_display = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# #Mostrando as imagens
plt.figure(figsize=[20,10])
plt.subplot(121); plt.axis('off'); plt.imshow(im1_display); plt.title('Foto Original')
plt.subplot(122); plt.axis('off'); plt.imshow(im2_display); plt.title('Foto Scanneada')

#Identificar pontos de referência entre as duas imagens
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

#Ordenar encontros por score
matches = sorted(list(matches), key=lambda x: x.distance, reverse=False)

#Remover encontros muito distantes
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

#Desenhar melhores encontros
im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

plt.figure(figsize=[40,10])
plt.imshow(im_matches); plt.axis('off'); plt.title("Encontros")

#Econtrar Homography
#Extrair localização dos encontros coerentes
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

#Homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

#Transformar a imagem Scanneada
#Usar Homography para transformar a imagem
height, width, channels = im1.shape
im2_reg = cv2.warpPerspective(im2, h, (width,height))

#Mostrar resultados
plt.figure(figsize=[20,10])
plt.subplot(121); plt.imshow(im1); plt.axis('off'); plt.title('Foto Original')
plt.subplot(122); plt.imshow(im2_reg); plt.axis('off'); plt.title('Foto Scanneada')
plt.show()