import os
import glob
from IPython import embed 
import re
from loguru import logger
class Market1501():
    """
    """
    dataset_dir = 'Market-1501-v15.09.15'
    def __init__(self,root='/media/data/dataset',**kwargs):
        self.dataset_dir = os.path.join(root,self.dataset_dir)
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
    
if __name__=='__main__':
    data =  Market1501()


