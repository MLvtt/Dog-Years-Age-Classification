import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class DogFaceDetector:
    def __init__(self, det_path: str or bool or None='../data/dogHeadDetector.dat', 
                    pred_path: str or bool or None='../data/landmarkDetector.dat') -> None:
        self.det_path = det_path
        self.pred_path = pred_path
        self.load_detector_predictor()

    def load_detector_predictor(self) -> None:
        self.detector, self.predictor = None, None
        if (self.det_path != False) and (self.det_path != None):
            tqdm.write('Loading Detector...')
            self.detector = dlib.cnn_face_detection_model_v1(self.det_path)
            tqdm.write('Detector Loaded.')
        if (self.pred_path != False) and (self.pred_path != None):
            tqdm.write('Loading Predictor...')
            self.predictor = dlib.shape_predictor(self.pred_path)
            tqdm.write('Predictor Loaded.')

    def get_dogface(self, img_path: str, dsize: tuple=(500,500), 
                    outpath: str or None or bool='../img/',
                    verbose: bool=True,
                    predict_features: bool=False,
                    # padding: int or None or bool=5, 
                    save: bool=True) -> dict or None:
        self.img_path = img_path
        self.dsize = dsize
        self.verbose = verbose
        # self.padding = padding

        # def padding_ratio():
        #     # if (self.padding != None) and (self.padding != False):
        #     #     h, w = self.img_og.shape[:2]
        #     #     self.x_pad, self.y_pad = int(padding * w/h), int(padding * h/w)
        #     # else:
        #     #     self.x_pad = self.y_pad = self.padding

        self._load_format_img()
        self._detect_faces()
        if self.face_detect_count == 0:
            return
        # padding_ratio()
        self._get_dogface_imgs(outpath, predict_features, save)
        return self.dogface_imgs

    def _get_dogface_imgs(self, outpath: str or None or bool, 
                            predict_features: bool, save: bool) -> None:
        self.img_result = self.img_og.copy()
        self.bbox_dims = {}
        self.dogface_imgs = {}
        self.face_features = {}

        for i, d in enumerate(self.faces_detected):
            ## Get Bbox Coords
            x1, y1, x2, y2 = self._get_bbox(d)
            # if (x1 - self.x_pad) < 0:
                # x1 = self.x_pad
            # if (y1 - self.y_pad) < 0:
                # y1 = self.y_pad
            
            ## Save to BBox dict
            self.bbox_dims[i] = (x1, y1, x2, y2)
            
            if self.verbose:
                tqdm.write(F"Detection {i}: Left: {x1} \
                                Top: {y1} Right: {x2} Bottom: {y2} \
                                Confidence: {d.confidence}")
            
                
            ## add bbox to result img
            cv2.rectangle(self.img_result, pt1=(x1, y1), pt2=(x2, y2), 
                            thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)
            dogface_img = self.img_og[y1:y2, x1:x2]
            if predict_features and (self.pred_path != False) and (self.pred_path != None):
                shape = self._detect_features(d)
                self.face_features[i] = shape

            dogface_img = cv2.cvtColor(dogface_img, cv2.COLOR_BGR2RGB)
            dogface_img = cv2.resize(dogface_img, dsize=(224,224), 
                                        fx=0.5, fy=0.5)
                
            # try:
            #     ## Get Dogface img w/w/o padding
            #     dogface_img = self.img_og[y1-self.y_pad:y2+self.y_pad, x1-self.x_pad:x2+self.x_pad]
            #     ## Resize to 224,224
            #     dogface_img = cv2.resize(dogface_img, dsize=(224,224), fx=0.5, fy=0.5)
            #     dogface_img = cv2.cvtColor(dogface_img, cv2.COLOR_BGR2RGB)
            # except cv2.error:
            #     dogface_img = self.img_og[y1:y2, x1:x2]
            #     dogface_img = cv2.resize(dogface_img, dsize=(224,224), fx=0.5, fy=0.5)
            #     dogface_img = cv2.cvtColor(dogface_img, cv2.COLOR_BGR2RGB)
                

            ## add to dogface imgs dict
            self.dogface_imgs[i] = dogface_img
            if save and ((outpath != None) and (outpath != False)):
                ## save dogface img   
                dogface_filename = f'{outpath}{self.filename}_face{i}{self.ext}'
                if self.verbose:
                    tqdm.write(f'Saving To {dogface_filename}')
                
                cv2.imwrite(dogface_filename, dogface_img)

        # for i, bbox in self.bbox_dims.items():
            # Create dogfaces
            # padding correct for resizing

    def check_for_face(self, img_path: str, dsize: tuple=(224,224)) -> int:
        self.img_path = img_path
        self.dsize = dsize
        self._load_format_img()
        self._detect_faces()
        return self.face_detect_count


    def _get_bbox(self, d) -> tuple:
        x1, y1, x2, y2 = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > self.img_og.shape[1]:
            x2 = self.img_og.shape[1]
        if y2 > self.img_og.shape[0]:
            y2 = self.img_og.shape[0]
        return (x1, y1, x2, y2)

    def _detect_faces(self) -> None:
        self.faces_detected = self.detector(self.img_og, upsample_num_times=1)
        self.face_detect_count = len(self.faces_detected)
        if self.face_detect_count == 0:
            if self.verbose:
                tqdm.write('Could Not Find Dog Face')
            # self._detect_faces()
        elif self.face_detect_count == 1:
            if self.verbose:
                tqdm.write('1 Dog Face Found')
        else:
            if self.verbose:
                tqdm.write(f'{self.face_detect_count} Dog Faces Found')
    
    def _detect_features(self, d):
        shape = self.predictor(self.img_og, d.rect)
        shape = face_utils.shape_to_np(shape)
        
        for i, p in enumerate(shape):
            cv2.circle(self.img_result, center=tuple(p), radius=3, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(self.img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return shape

    def _load_format_img(self) -> None:
        self.filename, self.ext = os.path.splitext(os.path.basename(self.img_path))
        self.img_og = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        img_og_shape = self.img_og.shape
        sizelim=800
        if (img_og_shape[0] > sizelim) or (img_og_shape[1] > sizelim):
            new_og_size = (int(sizelim * img_og_shape[1]/img_og_shape[0]), int(sizelim * img_og_shape[0]/img_og_shape[1]))
            self.img_og = cv2.resize(self.img_og, dsize=new_og_size)
        # self.img = cv2.resize(self.img_og, dsize=self.dsize, fx=0.5, fy=0.5)

 
def check_dogface_dataset(dataset_path):
    dfd = DogFaceDetector(pred_path=None)
    ds_name = dataset_path.split('/')[-1].split('_')[0].lower()
    filepaths = [fp for fp, cat, img_lst in os.walk(dataset_path)]
    categories = [cat for fp, cat, img_lst in os.walk(dataset_path)][0]
    img_lists = [img_lst for fp, cat, img_lst in os.walk(dataset_path)]
    no_face_detected = set()
    for fp, cat, img_lst in zip(filepaths[1:], categories, img_lists[1:]):
        tqdm.write(fp)
        pbar = tqdm(img_lst)
        no_face_counter = 0
        for i, fn in enumerate(pbar):
            if fn == '.DS_Store':
                continue
            if fp[-1] == '/':
                img_path = fp + fn
            else:    
                img_path = fp + '/' + fn
            tqdm.write(img_path)
            face_count = dfd.check_for_face(img_path)
            if face_count == 0:
                no_face_detected.add(img_path)
                tqdm.write(fn)
                no_face_counter += 1
            elif face_count > 1:
                tqdm.write(fn)
        tqdm.write(f"{cat} done - {no_face_counter} no face detected")
    tqdm.write(str(len(no_face_detected)))



if __name__ == '__main__':
    # img_path = '../../data/PetFinder_All/Senior/43690994_3.jpg'
    img_path = '/Users/mbun/Code/dsi_galvanize/capstones/capstone_3/data/test_dataset/0V6Z5H8KK0.jpg'
    dfd = DogFaceDetector()
    # dfd.img_path = img_path
    # dfd._load_format_img()
    # plt.imshow(dfd.img_og)
    df_img = dfd.get_dogface(img_path, predict_features=True, save=False)
    # print(dfd.img_og)
    plt.imshow(dfd.img_result)
    plt.show()
    # dfd.check_for_face(img_path)

    # DogFaceID(img_path)
    # img_path = '../../data/PetFinder_All/Adult/43953211_2.jpg'
    # print(cv2.imread(img_path).shape)
    # print(type(cv2.imread(img_path)))
    # plt.imshow(load_format_image(img_path)[0])
    # plt.imshow(cv2.imread(img_path))
    # plt.imshow(DogFaceOnly(img_path, dsize=(224,224)))
    # plt.show()
    # path = '../../data/PetFinder_All'

    # check_dogface_dataset(path)

    # dataset_path = '../../data/PetFinder_All'
    # filepaths = [x for x, y, z in os.walk(dataset_path)]
    # categories = [y for x, y, z in os.walk(dataset_path)][0]
    # img_lists = [z for x, y, z in os.walk(dataset_path)]

    # # print(filepaths)
    # # print(categories)
    # for fpath, cat, img_lst in zip(filepaths[1:], categories, img_lists[1:]):
    #     for fname in img_lst:
    #         if fname != '.DS_Store':
    #             dfd.get_dogface(img_path=fpath+'/'+fname, 
                                #   outpath=f"../../data/pf_face_dataset/{cat}/")
    # print(img_lists)