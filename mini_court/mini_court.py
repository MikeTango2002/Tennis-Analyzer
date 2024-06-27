import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (convert_meters_to_pixel_distance, 
                   convert_pixel_distance_to_meters, 
                   get_foot_position, 
                   get_closest_keypoint_index,
                   measure_xy_distance,
                   get_center_of_bbox,
                   measure_distance)

class MiniCourt(): #Classe per disegnare il mini campo da tennis su schermo
    def __init__(self, frame): #Il frame è il frame iniziale del video
        self.drawing_rectangle_width = 250 #Le misure sono in pixel
        self.drawing_rectangle_height = 500
        self.buffer = 70 #distanza dal bordo del frame
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()


    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        # calcolo le coordinate del rettangolo contenente il mini court all'interno del frame
        self.end_x = frame.shape[1] - (self.buffer + 20)
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height


    def set_mini_court_position(self):

        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court 
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court 

        self.court_drawing_width = self.court_end_x - self.court_start_x

    
    def convert_meters_to_pixels(self, meters): #Serve per ottenere uns conversione da metri a pixel del mini_court
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )


    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0]*28 #Inizializzo la lista con 28 elementi

        #Punto 0
        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)

        #Punto 1
        drawing_keypoints[2], drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)

        #Punto 2
        drawing_keypoints[4] = int(self.court_start_x)
        drawing_keypoints[5] = int(self.court_start_y) + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)

        #Punto 3
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width
        drawing_keypoints[7] = drawing_keypoints[5] 

        #Punto 4
        drawing_keypoints[8] = drawing_keypoints[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1] 

        #Punto 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5] 

        #Punto 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[3] 

        #Punto 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[15] = drawing_keypoints[7] 

        #Punto 8
        drawing_keypoints[16] = drawing_keypoints[8] 
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)

        #Punto 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17] 

        #Punto 10
        drawing_keypoints[20] = drawing_keypoints[10] 
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)

        #Punto 11
        drawing_keypoints[22] = drawing_keypoints[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21] 

        #Punto 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18])/2)
        drawing_keypoints[25] = drawing_keypoints[17] 

        #Punto 13
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22])/2)
        drawing_keypoints[27] = drawing_keypoints[21] 

        self.drawing_key_points=drawing_keypoints


    def set_court_lines(self):

        #Tuple contententi le coppie di punti che definiscono le linee del campo
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            (0,1),
            (8,9),
            (10,11),
            (12,13),
            (2,3)
        ]

    def draw_background_rectangle(self, frame):
        
        shapes = np.zeros_like(frame, np.uint8) #Crea un'immagine nera delle stesse dimensioni del frame

        #Disegna il rettangolo di sfondo
        cv2.rectangle(shapes, (self.start_x, self.start_y - 40), (self.end_x, self.end_y + 40), (255, 255, 255), cv2.FILLED)

        out = frame.copy()
        alpha = 0.5 #Trasparenza del background

        #Viene creata una maschera booleana dalla matrice shapes. 
        #Questo significa che ogni pixel che è diverso da zero in shapes sarà considerato True nella maschera.
        #Sono considerati True solo i pixel bianchi del rettangolo
        mask = shapes.astype(bool) 

        #La seguente funzione  viene utilizzata per combinare il frame originale e l'immagine shapes 
        #con i coefficienti di trasparenza specificati. Il risultato è che il rettangolo bianco 
        #viene disegnato sul frame originale con una trasparenza del 50%. 
        #La maschera viene utilizzata per applicare questa combinazione solo nelle aree del rettangolo.
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out
    
    def draw_court(self, frame):

        #Vengono prima disegnati i keypoints del campo
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        #Poi vengono disegnate le linee del campo
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2 + 1])) #il "*2" è dovuto a come sono memorizzate le coordinate nelle tuple di self.lines
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        #Infine si disegna il net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    
    def draw_mini_court(self, frames):#Funzione che racchiude tutte le funzioni di disegno del mini campo
        
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame) 

        return output_frames
    

    def get_start_point_of_mini_court(self):
        return(self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_keypoints(self):
        return self.drawing_key_points
    
    def compute_homography_matrix(self, original_keypoints): #Calcola la matrice omografica per trasformare le coordinate del campo di tennis in coordinate del mini campo

        original_court_keypoints = original_keypoints.copy()
        #Trasformo l'array 1d in un array 2d
        num_points = original_court_keypoints.shape[0] // 2
        original_court_keypoints = original_court_keypoints.reshape((num_points, 2))


        transformed_court_keypoints = np.array(self.get_court_keypoints(), dtype='float32').copy()
        #Trasformo l'array 1d in un array 2d
        transformed_court_keypoints = transformed_court_keypoints.reshape((num_points, 2))


        self.H, _ = cv2.findHomography(original_court_keypoints, transformed_court_keypoints)


    
    def get_mini_court_coordinates(self, object_position):
        #In questa funzione, converto le coordinate di un oggetto (pallina o giocatore) rispetto al campo di tennis in coordinate rispetto al mini campo
        #Per fare ciò utilizzo la matrice omografica calcolata in precedenza

        # Trasforma le posizioni dei giocatori
        x, y = object_position

        original_point_homogeneous = np.array([x, y, 1], dtype='float32') #Aggiunge la coordinata z=1 per rendere la posizione omogenea

        transformed_point_homogeneous = np.dot(self.H, original_point_homogeneous)

        # Normalizza le coordinate trasformate
        transformed_point = transformed_point_homogeneous / transformed_point_homogeneous[2]

        # Ottieni le coordinate cartesiane trasformate
        transformed_x, transformed_y = transformed_point[:2]

        return (transformed_x, transformed_y)
        
        

    def convert_bounding_boxes_to_mini_court_coordinates(self, players_boxes, ball_boxes, original_court_keypoints, frames_with_ball_hit): #Converte le coordinate dei bounding boxes dei giocatori e della pallina in coordinate del mini campo

        self.compute_homography_matrix(original_court_keypoints)



        output_players_boxes = []
        output_ball_boxes = []

        ball_bounce_num = 0
        current_output_ball_boxes_dict = {}
        
        for frame_num, players_bbox in enumerate(players_boxes):


            current_output_ball_boxes = current_output_ball_boxes_dict.copy()


            if frame_num in frames_with_ball_hit:
                
                #Calcolo delle coordinate della pallina
                ball_box = ball_boxes[frame_num][1]
                ball_position = get_center_of_bbox(ball_box)
                player_id_shot = 1
                min_distance = 1000000
                position = (0,0)
                for player_id, bbox in players_bbox.items():
                    foot_position = get_foot_position(bbox)
                    distance = measure_distance(ball_position, foot_position)
                    if distance < min_distance:
                        player_id_shot = player_id
                        position = foot_position
                        min_distance = distance
                mini_court_ball_position = self.get_mini_court_coordinates(position)
                
                current_output_ball_boxes_dict[ball_bounce_num]= mini_court_ball_position
                ball_bounce_num = ball_bounce_num + 1

            current_output_ball_boxes = current_output_ball_boxes_dict.copy()
        
            output_ball_boxes.append(current_output_ball_boxes)

            #Calcolo delle coordinate dei giocatori
            output_players_bboxes_dict = {}

            for player_id, bbox in players_bbox.items():

                foot_position = get_foot_position(bbox)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position) 
                                                                             
                output_players_bboxes_dict[player_id] = mini_court_player_position    

            output_players_boxes.append(output_players_bboxes_dict)

        return output_players_boxes, output_ball_boxes
    

    def draw_points_on_mini_court(self, frames, positions, color=(0,255,0)): #Disegna i keypoints sul mini campo
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                if len(positions[frame_num]) == 0:
                    continue
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames
