import os
import glob
from IPython import embed 
import re
from loguru import logger
class Market1501():
    """
    """

    def __init__(self,root='/media/data/dataset',dataset_dir = 'Market-1501-v15.09.15',**kwargs):
        self.dataset_dir = os.path.join(root,dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir,'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir,'query')
        self.gallery_dir = os.path.join(self.dataset_dir,'bounding_box_test')
        self._check_dataset()
        train, trian_num_pids, train_num_imgs,train_relabeled = self._process_directory(self.train_dir,relabel=True)
        query, query_num_pids, query_num_imgs,query_relabeled = self._process_directory(self.query_dir,relabel=False)
        gallery, gallery_num_pids, gallery_num_imgs,gallery_relabeled = self._process_directory(self.gallery_dir,relabel=False)
        
        total_num_pids = trian_num_pids + query_num_pids + gallery_num_pids
        total_num_imgs = train_num_imgs + query_num_imgs + gallery_num_imgs
        
        logger.info("\n\n\n => Market1501 loaded\n")
        logger.info("--------------------------------------")
        logger.info("  subset  |  ids | images | relabeled")
        logger.info("--------------------------------------")
        logger.info("  train   |{:5d} |{:8d}|   {}  ".format(trian_num_pids,train_num_imgs,train_relabeled))
        logger.info("  query   |{:5d} |{:8d}|   {} ".format(query_num_pids,query_num_imgs,query_relabeled))
        logger.info("  gallery |{:5d} |{:8d}|   {} ".format(gallery_num_pids,gallery_num_imgs,gallery_relabeled))
        logger.info("--------------------------------------")
        logger.info("  total   |{:5d} |{:8d}|     - ".format(total_num_pids,total_num_imgs))
        logger.info("--------------------------------------")
        logger.info('\n\n')


        self.train, self.query, self.gallery = train, query, gallery
        self.train_num_pids, self.query_num_pids, self.gallery_num_pids = trian_num_pids, query_num_pids, gallery_num_pids
        self.train_num_imgs, self.query_num_imgs, self.gallery_num_imgs = train_num_imgs, query_num_imgs, gallery_num_imgs
    
    def _check_dataset(self):
        def check_didrctory_existence(dir_path):
            if not os.path.exists(dir_path) :
                raise RuntimeError(f"{dir_path} does not exist.")
        check_didrctory_existence(self.dataset_dir)
        check_didrctory_existence(self.train_dir)
        check_didrctory_existence(self.query_dir)
        check_didrctory_existence(self.gallery_dir)

    def _process_directory(self, dir_path,relabel = False):
        img_paths = glob.glob(os.path.join(dir_path,"*.jpg")) # Return a list of paths matching the pattern '*jpg'
        pattern = re.compile(r'([-\d]+)_c(\d)')
        
        pid_set = set()
        dataset = []
        for img_path in img_paths:
            m = pattern.search(img_path)
            if m is None:
                continue
            pid,camid = map(int,m.groups())
            if pid == -1: 
                continue
            assert 0<=pid<=1501
            assert 1<=camid<=6
            pid_set.add(pid)
            dataset.append((img_path, pid, camid))
        if relabel:
            pid2label = {pid:label for label,pid in enumerate(pid_set)}
            dataset = [(img_path, pid2label[pid],camid-1) for img_path,pid,camid in dataset]
        num_pids = len(pid_set)
        num_imgs = len(img_paths) 
        # embed()
        return dataset, num_pids, num_imgs, relabel
    
class MSMT17_V1():
    """
    """

    def __init__(self,root='/media/data/dataset',dataset_dir = 'MSMT17_V1',**kwargs):
        self.dataset_dir = os.path.join(root,dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.test_dir = os.path.join(self.dataset_dir, 'test')
        self.list_train_path = os.path.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = os.path.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = os.path.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = os.path.join(self.dataset_dir, 'list_gallery.txt')

        self._check_dataset()
        train, trian_num_pids, train_num_imgs,train_relabeled = self._process_directory(self.train_dir,self.list_train_path,relabel=True)
        query, query_num_pids, query_num_imgs,query_relabeled = self._process_directory(self.test_dir,self.list_query_path,relabel=False)
        gallery, gallery_num_pids, gallery_num_imgs,gallery_relabeled = self._process_directory(self.test_dir,self.list_gallery_path,relabel=False)
        val,val_num_pids,val_num_imgs,val_relabeled = self._process_directory(self.test_dir,self.list_val_path,relabel=False)
        total_num_pids = trian_num_pids + query_num_pids + gallery_num_pids + val_num_imgs
        total_num_imgs = train_num_imgs + query_num_imgs + gallery_num_imgs + val_num_pids
        
        logger.info("\n\n\n => MSMT17_V1 loaded\n")
        logger.info("--------------------------------------")
        logger.info("  subset  |  ids | images | relabeled")
        logger.info("--------------------------------------")
        logger.info("  train   |{:5d} |{:8d}|   {}  ".format(trian_num_pids,train_num_imgs,train_relabeled))
        logger.info("  val     |{:5d} |{:8d}|   {}  ".format(val_num_pids,val_num_imgs,val_relabeled))
        logger.info("  query   |{:5d} |{:8d}|   {} ".format(query_num_pids,query_num_imgs,query_relabeled))
        logger.info("  gallery |{:5d} |{:8d}|   {} ".format(gallery_num_pids,gallery_num_imgs,gallery_relabeled))
        logger.info("--------------------------------------")
        logger.info("  total   |{:5d} |{:8d}|     - ".format(total_num_pids,total_num_imgs))
        logger.info("--------------------------------------")
        logger.info('\n\n')


        self.train, self.query, self.gallery,self.val = train, query, gallery, val
        self.train_num_pids, self.query_num_pids, self.gallery_num_pids, self.val_num_pids = trian_num_pids, query_num_pids, gallery_num_pids, val_num_pids
        self.train_num_imgs, self.query_num_imgs, self.gallery_num_imgs, self.val_num_imgs = train_num_imgs, query_num_imgs, gallery_num_imgs, val_num_imgs
    
    def _check_dataset(self):
        def check_didrctory_existence(dir_path):
            if not os.path.exists(dir_path) :
                raise RuntimeError(f"{dir_path} does not exist.")
        check_didrctory_existence(self.dataset_dir)
        check_didrctory_existence(self.train_dir)
        check_didrctory_existence(self.test_dir)
        check_didrctory_existence(self.list_gallery_path)
        check_didrctory_existence(self.list_train_path)
        check_didrctory_existence(self.list_val_path)
        check_didrctory_existence(self.list_query_path)


    def _process_directory(self, dir_path,list_path,relabel = False):
        # img_paths = glob.glob(os.path.join(dir_path,"*.jpg")) # Return a list of paths matching the pattern '*jpg'
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        pid_set = set()
        dataset = []
        cam_set = []
        for img_idx,img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)
            camid = int(img_path.split('_')[2])
            img_path  = os.path.join(dir_path,img_path)
            pid_set.add(pid)
            cam_set.append(camid)
            dataset.append((img_path, pid, camid))
        if relabel:
            pid2label = {pid:label for label,pid in enumerate(pid_set)}
            dataset = [(img_path, pid2label[pid],camid-1) for img_path,pid,camid in dataset]
        num_pids = len(pid_set)
        num_imgs = len(dataset) 
        # embed()
        return dataset, num_pids, num_imgs, relabel


class OccludedDuke():
    """
    """

    def __init__(self,root='/media/data/dataset',dataset_dir = 'Occluded_Duke',**kwargs):
        self.dataset_dir = os.path.join(root,dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir,'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir,'query')
        self.gallery_dir = os.path.join(self.dataset_dir,'bounding_box_test')
        self._check_dataset()
        train, trian_num_pids, train_num_imgs,train_relabeled = self._process_directory(self.train_dir,relabel=True)
        query, query_num_pids, query_num_imgs,query_relabeled = self._process_directory(self.query_dir,relabel=False)
        gallery, gallery_num_pids, gallery_num_imgs,gallery_relabeled = self._process_directory(self.gallery_dir,relabel=False)
        
        total_num_pids = trian_num_pids + query_num_pids + gallery_num_pids
        total_num_imgs = train_num_imgs + query_num_imgs + gallery_num_imgs
        
        logger.info("\n\n\n => Occluded_Duke loaded\n")
        logger.info("--------------------------------------")
        logger.info("  subset  |  ids | images | relabeled")
        logger.info("--------------------------------------")
        logger.info("  train   |{:5d} |{:8d}|   {}  ".format(trian_num_pids,train_num_imgs,train_relabeled))
        logger.info("  query   |{:5d} |{:8d}|   {} ".format(query_num_pids,query_num_imgs,query_relabeled))
        logger.info("  gallery |{:5d} |{:8d}|   {} ".format(gallery_num_pids,gallery_num_imgs,gallery_relabeled))
        logger.info("--------------------------------------")
        logger.info("  total   |{:5d} |{:8d}|     - ".format(total_num_pids,total_num_imgs))
        logger.info("--------------------------------------")
        logger.info('\n\n')


        self.train, self.query, self.gallery = train, query, gallery
        self.train_num_pids, self.query_num_pids, self.gallery_num_pids = trian_num_pids, query_num_pids, gallery_num_pids
        self.train_num_imgs, self.query_num_imgs, self.gallery_num_imgs = train_num_imgs, query_num_imgs, gallery_num_imgs
    
    def _check_dataset(self):
        def check_didrctory_existence(dir_path):
            if not os.path.exists(dir_path) :
                raise RuntimeError(f"{dir_path} does not exist.")
        check_didrctory_existence(self.dataset_dir)
        check_didrctory_existence(self.train_dir)
        check_didrctory_existence(self.query_dir)
        check_didrctory_existence(self.gallery_dir)

    def _process_directory(self, dir_path,relabel = False):
        img_paths = glob.glob(os.path.join(dir_path,"*.jpg")) # Return a list of paths matching the pattern '*jpg'
        pattern = re.compile(r'([-\d]+)_c(\d)')
        
        pid_set = set()
        dataset = []
        for img_path in img_paths:
            m = pattern.search(img_path)
            if m is None:
                continue
            pid,camid = map(int,m.groups())
            if pid == -1: 
                continue
            # assert 0<=pid<=1501
            # assert 1<=camid<=6
            pid_set.add(pid)
            dataset.append((img_path, pid, camid))
        if relabel:
            pid2label = {pid:label for label,pid in enumerate(pid_set)}
            dataset = [(img_path, pid2label[pid],camid-1) for img_path,pid,camid in dataset]
        num_pids = len(pid_set)
        num_imgs = len(img_paths) 
        # embed()
        return dataset, num_pids, num_imgs, relabel



if __name__=='__main__':
    data = MSMT17_V1()


