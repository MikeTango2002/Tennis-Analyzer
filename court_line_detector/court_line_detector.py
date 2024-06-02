import torch 
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
import pickle

class CourtLineDetector:

    def __init__(self, model_path):

        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')) #carica i pesi addestrati precedentemente e assegna questi pesi al modello corrente

        self.transform = transforms.Compose([ #Lista delle trasformazioni da applicare all'immagine (sono le stesse del file di training)
            transforms.ToPILImage(), #Trasforma l'immagine in un oggetto PIL (Python Imaging Library)
            transforms.Resize((224, 224)), #Ridimensiona l'immagine a 224x224 pixel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # sono le medie e le deviazioni standard dei canali di colore rosso, verde e blu (RGB) calcolate su un grande dataset di immagini, comunemente il dataset ImageNet.
        ])

    def predict_multiple_frames(self, frames, read_from_stub=False, stub_path=None, edge_detection=False, window_size=30, num_frames=3):

        keypoints_detections = [] #E' una lista di array numpy, dove ogni array numpy contiene le keypoints detection di un frame

        if read_from_stub and stub_path is not None: 
            with open(stub_path, 'rb') as f:
                keypoints_detections = pickle.load(f)
            return keypoints_detections
        
        #Calcolo i keypoints come la media dei keypoints dei primi num_frames frame, in quanto la videocamera è fissa e i keypoints non dovrebbero variare molto
        keypoints_detections_temp = [] 
        
        for i in range(num_frames):
            keypoints_dict = self.predict_frame(frames[i], window_size, edge_detection)
            keypoints_detections_temp.append(keypoints_dict)

        keypoints = keypoints_detections_temp[0] #Inizializzo le keypoints con quelle dell'ultimo frame
        for i in range(1, num_frames):
            keypoints = keypoints + keypoints_detections_temp[i] 

        keypoints = keypoints / num_frames

        #Ora assegno i keypoints appena ottenuti a tutti i frame del video
        for frame in frames: 
            keypoints_detections.append(keypoints)

        if stub_path is not None: #Se è stato specificato un percorso per il file stub, salva le detections in un file stub
            with open(stub_path, 'wb') as f:
                pickle.dump(keypoints_detections, f)

        return keypoints_detections
    
    def predict_frame(self, image, window_size, edge_detection=False):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converte l'immagine da BGR a RGB
        image_tensor = self.transform(img_rgb).unsqueeze(0) #Applica le trasformazioni all'immagine e aggiunge una dimensione all'inizio del tensore per renderlo compatibile con il modello che prende in input un batch (lista) di immagini

        with torch.no_grad(): #Poichè stiamo facendo inferenza, non abbiamo bisogno di calcolare i gradienti
            output = self.model(image_tensor)
        
        keypoints = output.squeeze().cpu().numpy() 


        original_h, original_w = img_rgb.shape[:2] #Prende le dimensioni originali dell'immagine
        
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        if edge_detection:
            # Ora che abbiamo le keypoints, possiamo applicare il post-processing per centrare le keypoints intorno ai contorni
            for i in range(0, len(keypoints), 2):

                x = int(keypoints[i])
                y = int(keypoints[i + 1])

                keypoint = (x, y)

                # Prima si definiscono i limiti della finestra intorno al keypoint
                top_left_x = max(0, x - window_size)
                top_left_y = max(0, y - window_size)
                bottom_right_x = min(image.shape[1], x + window_size)
                bottom_right_y = min(image.shape[0], y + window_size)

                # Estrai la regione di interesse (ROI) intorno al keypoint
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                roi = gray[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                # Applica il filtro di Canny per l'edge detection sulla ROI
                edges = cv2.Canny(roi, threshold1=150, threshold2=50)
                #Soglia alta (threshold1): Qualsiasi pixel con un gradiente maggiore di questa soglia viene immediatamente classificato come un bordo. Questa soglia è spesso chiamata "soglia alta".
                #Soglia bassa (threshold2): I pixel con un gradiente inferiore a questa soglia vengono ignorati e non sono considerati bordi. Questa soglia è spesso chiamata "soglia bassa".
                #Isteresi: I pixel con un gradiente tra threshold1 e threshold2 sono classificati come bordi solo se sono collegati a un pixel che è già stato classificato come bordo (ovvero, un pixel con un gradiente maggiore di threshold1).

                # Trova i contorni nella ROI
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Trova il contorno più grande
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Trova il centro del contorno più grande
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + top_left_x
                        cy = int(M["m01"] / M["m00"]) + top_left_y
                        centered_keypoint = (cx, cy)
                    else:
                        centered_keypoint = keypoint  # Se non riesce a trovare il centro, mantieni le coordinate originali del keypoint
                else:
                    centered_keypoint = keypoint  # Se non ci sono contorni, mantieni le coordinate originali del keypoint

                keypoints[i] = centered_keypoint[0]
                keypoints[i + 1] = centered_keypoint[1]
        #Fine edge detection

        return keypoints
    
    
    def draw_keypoints(self, image, keypoints): #Disegna i keypoints sull'immagine


        for i in range(0, len(keypoints), 2): #Per ogni coppia di coordinate x, y
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.putText(image, str(i // 2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1) #-1 indica che il cerchio sarà pieno
        
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):

        output_video_frames = []

        for frame, keypoints_frame in zip(video_frames, keypoints):

            output_frame = self.draw_keypoints(frame, keypoints_frame)
            output_video_frames.append(output_frame)
        
        return output_video_frames