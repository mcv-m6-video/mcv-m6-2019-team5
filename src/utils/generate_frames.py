import cv2


def generate_frames(vidcap, folder):
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(folder+"/frame%04d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
