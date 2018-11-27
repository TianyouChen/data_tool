import cv2
import os
def save_img():
    video_path = r'/home/ubuntu/facial_video/S4/6_Neutral-Surprise/'
    videos = os.listdir(video_path)
    print videos
    for video_name in videos:
        file_name = video_name.split('.')[0]
        print file_name
        folder_name = '/home/ubuntu/Save_img/' + file_name
        os.mkdir(folder_name)
        vc = cv2.VideoCapture(video_path+video_name)
        c=0
        rval=vc.isOpened()

        while rval:   
            c = c + 1
            rval, frame = vc.read()
            pic_path = folder_name+'/'
            if rval:
                cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)
                cv2.waitKey(1)
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)
save_img()
