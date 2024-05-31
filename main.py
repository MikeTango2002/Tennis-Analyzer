from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker


def main():
    #Read Video
    input_video_path = "input_videos/input.mp4"
    video_frames = read_video(input_video_path)

    #Detect Players
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    player_detections = player_tracker.detect_multiple_frames(video_frames)

    #Draw output

    ##Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)


    save_video(video_frames, "output_videos/output.avi")

    if __name__ == '__main__':
        main()