
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2) #int perchè sono pixel
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoints_indices_to_consider):

    #Inizializza le variabili
    closest_distance = float('inf')
    key_point_index_closest = keypoints_indices_to_consider[0]

    for keypoint_index in keypoints_indices_to_consider:
        keypoint = keypoints[keypoint_index*2], keypoints[keypoint_index*2 + 1] 
        distance = abs(point[1] - keypoint[1]) #Considero solo la distanza in y tra il punto e il keypoint
        if distance < closest_distance:
            closest_distance = distance
            key_point_index_closest = keypoint_index
    
    return key_point_index_closest

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)