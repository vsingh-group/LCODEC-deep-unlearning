'''
BE VERY CAREFUL RUNNING THIS SCRIPT.

MAKES MANY HTTP REQUESTS PER SECOND,
MAY TRIP FIREWALLS FOR CONNECTING TO 
RANDOM/UNKNOWN/SUSPICIOUS SERVERS.

!!!!!!!!!!!! PLEASE BE CAREFUL !!!!!!

'''
import pandas as pd
from pathlib import Path
import time

import os
import sys
import threading
import urllib.request
import socket
from queue import Queue
from PIL import Image

socket.setdefaulttimeout(5)
n_threads = 48

# pathlist = sorted(Path("files").glob('**/*.txt'))
columns = ['ID', 'URL', 'BBL', 'BBT', 'BBR', 'BBB', 'Pose', 'DPMScore', 'Curation']
outcsvname = '100_home_dled_vggface.csv'
download_folder = '100_downloads/'
processed_folder = '100_images/'
namesinorder = 'names.txt'
max_imgs_per_person = 100

badls = 0

class CustomError(Exception):
    pass

class Downloader(threading.Thread):
    def __init__(self, queue):
        super(Downloader, self).__init__()
        self.queue = queue

    def run(self):

        global outcsvname
        global download_folder
        global processed_folder

        while True:
            i, img, person_name, pid = self.queue.get()

            # sentinel
            if person_name is None:
                queue.task_done()
                return

            download_dir = download_folder + person_name + '/'  
            processed_dir = processed_folder + person_name + '/'  

            #if len(os.listdir(processed_dir)) > max_imgs_per_person:
            #    with queue.mutex:
            #        queue.queue.clear()
            #    return

            #print(f'\tProcessing: {i}')
            url = img['URL'].split('.')

            filename = person_name + '_' + str(img['ID']) + '.' + url[-1]

            download_path = download_dir + filename
            processed_path = processed_dir + filename

            # print(url, '\t\t\t', processed_path)

            # only work with jpegs, cleaner
            if (not os.path.exists(download_path)) \
                    & (url[-1] == 'jpg') \
                    & (len(os.listdir(processed_dir)) <= max_imgs_per_person):
                try:
                    #urllib.request.urlretrieve(img['URL'], filename=download_path)
                    #urllib.request.urlopen(img['URL'], filename=download_path, timeout=5)
                    writefile = open(download_path,'wb')
                    page = urllib.request.urlopen(img['URL'], timeout=5).read()
                    writefile.write(page)
                    writefile.close()
                    #print('download success')

                    pilimg = Image.open(download_path)
                    # grayscale jpeg
                    if len(pilimg.getbands()) < 3:
                        raise CustomError("Not 3 channels, probably grayscale")

                    # face cropping
                    area = (img['BBL'], img['BBT'], img['BBR'], img['BBB'])
                    pilimg = pilimg.crop(area)

                    # resizing
                    width, height = pilimg.size
                    scalefactor = 256.0 / min([width, height])
                    pilimg = pilimg.resize((int(scalefactor*width), int(scalefactor*height)), resample=Image.BILINEAR)
                    # print(pilimg.size)

                    # center 224 cropping
                    w, h = pilimg.size
                    cx = int(w/2)
                    cy = int(h/2)
                    area = (cx-112, cy-112, cx+112, cy+112)
                    pilimg = pilimg.crop(area)

                    pilimg.save(processed_path)
                    # print('proc success')

                    img['Target'] = pid
                    img['Person'] = person_name
                    img['DownloadPath'] = download_path
                    img['FilePath'] = processed_path
                    img['FileName'] = filename
                    img['MyCuration'] = 1
                    img.to_frame().T.to_csv(outcsvname, sep=',', mode='a', index=False, header=not os.path.exists(outcsvname))

                    #print('\tProcessed:', img['URL'])

                except Exception as err:
                    #print("\tError processing %s: %s" % (img['URL'], err))
                    global badls
                    badls += 1


            queue.task_done()

if __name__ == '__main__':


    namesdf = pd.read_csv(namesinorder, header=None)

    cnt = 0
    for pid, name in namesdf.iterrows():
        person_name = name[0]

    # for path in pathlist:
        path = 'files/' + person_name + '.txt'
        imglist = pd.read_csv(path, delimiter=' ', header=None, names=columns)

        download_dir = download_folder + person_name + '/'  
        processed_dir = processed_folder + person_name + '/'  

        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)
        if not os.path.exists(download_dir):
            os.mkdir(download_dir)

        queue = Queue()
        threads = []
        for t in range(n_threads):
            threads.append(Downloader(queue))
            threads[-1].start()

        # person_name = path.__str__().split('/')[-1][:-4]

        for _, img in imglist.iterrows():
            queue.put((cnt, img, person_name, pid))
            cnt += 1

        print(f"Finishing current downloads for {person_name}...")
        for t in range(n_threads):
            queue.put((None, None, None, None))

        tmp = 0
        while not queue.empty():
            print('queue not empty, probably finishing up for ', person_name, ': ', tmp, 'secs, queue size: ', queue.qsize())
            tmp += 1
            time.sleep(1)

        print('Finished person: %s' % (person_name))



print(f'bad dls: {badls}')
