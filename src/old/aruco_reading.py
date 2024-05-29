import cv2
import numpy as np

# Carregar o vídeo
video_path = '../../rosbags/data/aruco_input_test.mp4'
cap = cv2.VideoCapture(video_path)

# Inicializar o detector de arucos
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

# Loop através dos frames do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar arucos no frame
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Desenhar caixas delimitadoras ao redor dos arucos detectados
    if ids is not None:
        for i in range(len(ids)):
            # Aumente a espessura da linha para 4 (por exemplo)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 0, 255))

            # # Extrair os cantos do marcador
            # marker_corners = corners[i][0]

            # # Desenhar um contorno ao redor do marcador
            # cv2.polylines(frame, [np.int32(marker_corners)], True, (0, 255, 0), thickness=2)

    # Mostrar o frame com os arucos detectados
    cv2.imshow('Frame', frame)

    # Sair do loop quando 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libertar os recursos
cap.release()
cv2.destroyAllWindows()



