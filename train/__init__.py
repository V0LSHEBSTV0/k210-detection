'''
    train main class

    @author neucrack@sipeed
    @license Apache 2.0 © 2020 Sipeed Ltd
'''

import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(root_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import Detector
import shutil
from utils import gpu_utils, isascii
from utils.logger import Logger
from instance import config
from datetime import datetime
import subprocess
import traceback
from enum import Enum

class TrainType(Enum):
    CLASSIFIER = 0
    DETECTOR   = 1

class TrainFailReason(Enum):
    ERROR_NONE     = 0
    ERROR_INTERNAL = 1
    ERROR_DOWNLOAD_DATASETS = 2
    ERROR_NODE_BUSY = 3
    ERROR_PARAM     = 4
    ERROR_CANCEL    = 5

class Train():
    def __init__(self,
                 datasets_zip,
                 datasets_dir,
                 out_dir):
        '''
            creat /temp/train_temp dir to train
        '''
        self.datasets_zip_path = datasets_zip

        self.temp_dir = out_dir

        assert os.path.exists(datasets_zip)
        self.datasets_dir = datasets_dir
        self.temp_datasets_dir = os.path.join(self.temp_dir, "datasets")
        self.result_dir = os.path.join(self.temp_dir, "result")
        self.clean_temp_files()
        os.makedirs(self.temp_dir)
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir)
        self.dataset_sample_images_path = os.path.join(self.temp_dir, "sample_images")
        os.makedirs(self.dataset_sample_images_path)
        self.log_file_path = os.path.join(self.temp_dir, "train_log.log")
        self.result_report_img_path = os.path.join(self.result_dir, "report.jpg")
        self.result_kmodel_path = os.path.join(self.result_dir, "m.kmodel")
        self.result_labels_path  = os.path.join(self.result_dir, "labels.txt")
        self.result_boot_py_path = os.path.join(self.result_dir, "boot.py")
        self.tflite_path = os.path.join(self.temp_dir, "m.tflite")
        self.final_h5_model_path = os.path.join(self.temp_dir, "m.h5")
        self.best_h5_model_path  = os.path.join(self.temp_dir, "m_best.h5")
        self.detector = None
        self.log = Logger(file_path=self.log_file_path)
    
    def __del__(self):
        # self.clean_temp_files()
        pass
    
    def clean_temp_files(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    def __on_progress(self, percent, msg):  # flag: progress
        self.log.i(f"progress: {percent}%, {msg}")
    
    def __on_fail(self, reson, msg, warn):
        self.log.e(f"failed: {reson}, {msg}")
        if warn:
            self.log.w(f"warnings:\n {warn}")

    def __on_train_progress(self, percent, msg):  # flag: progress
        percent = percent*0.97 + 1
        self.log.i(f"progress: {percent}%, {msg}")

    def train(self):
        warning_msg = ""
        try:
            warning_msg = self.train_process(self.log)
        except Exception as e:
            info = e.args[0]
            if type(info) == tuple and len(info) == 2:
                reason = info[0]
                msg = info[1]
                self.__on_fail(reason, msg, warning_msg)
            else:
                self.__on_fail(TrainFailReason.ERROR_INTERNAL, "node error:{}".format(e), warning_msg)

    def train_process(self, log):
        '''
            raise Exception if error occurred, a tuple: (TrainFailReason, error_message)
            @return result url
        '''

        # logging 
        self.__on_progress(0, "start") 
        self.__on_progress(1, "start train")        
        

        obj, prefix = self.detector_train(log = log)
        

        # check warnings
        result_warning_msg = ""
        result_warning_msg_path = os.path.join(self.result_dir, "warning.txt")
        if len(obj.warning_msg) > 0:
            result_warning_msg += "=========================================================================\n"
            result_warning_msg += "train warnings: these warn info may lead train error(accuracy loss), please check carefully\n"
            result_warning_msg += "=========================================================================\n\n\n"
            for msg in obj.warning_msg:
                result_warning_msg += "{}\n\n".format(msg)
            with open(result_warning_msg_path, "w") as f:
                f.write(result_warning_msg)

        # progress 99%
        self.__on_progress(99, "pack ok") # flag: progress

        # complete
        self.__on_progress(100, "task complete") # flag: progress
        return result_warning_msg

    def detector_train(self, log):
        
        # connect to GPU
        try:
            gpu = gpu_utils.select_gpu(memory_require = config.detector_train_gpu_mem_require, tf_gpu_mem_growth=False)
        except Exception:
            gpu = None
        if gpu is None:
            if not config.allow_cpu:
                log.e("no free GPU")
                raise Exception((TrainFailReason.ERROR_NODE_BUSY, "node no enough GPU or GPU memory and not support CPU train"))
            log.i("no GPU, will use [CPU]")
        else:
            log.i("select", gpu)


        # construcitng Detector 
        try:
            detector = Detector(input_shape=(224, 224, 3),
                                datasets_zip=self.datasets_zip_path,
                                datasets_dir=self.datasets_dir,
                                unpack_dir = self.temp_datasets_dir,
                                logger=log,
                                max_classes_limit = config.detector_train_max_classes_num,
                                one_class_min_images_num=config.detector_train_one_class_min_img_num,
                                one_class_max_images_num=config.detector_train_one_class_max_img_num,
                                allow_reshape = False)
            self.detector = detector
        except Exception as e:
            log.e("train datasets not valid: {}".format(e))
            raise Exception((TrainFailReason.ERROR_PARAM, "datasets not valid: {}".format(str(e))))
 
        # Trainging detector
        try:
            detector.train(epochs=config.detector_train_epochs,
                    progress_cb=self.__on_train_progress,
                    save_best_weights_path = self.best_h5_model_path,
                    save_final_weights_path = self.final_h5_model_path,
                    jitter=False,
                    is_only_detect = False,
                    batch_size = config.detector_train_batch_size,
                    train_times = 5,
                    valid_times = 2,
                    learning_rate=config.detector_train_learn_rate,
                )
        except Exception as e:
            log.e("train error: {}".format(e))
            traceback.print_exc()
            raise Exception((TrainFailReason.ERROR_INTERNAL, "error occurred when train, error: {}".format(str(e)) )) 
        
        # On good
        log.i("train ok, now generate report")
        detector.report(self.result_report_img_path)

        # Geterating kmodel
        log.i("now generate kmodel")
        detector.save(tflite_path=self.tflite_path)
        detector.get_sample_images(config.sample_image_num, self.dataset_sample_images_path)
        
        log.i("copy template files")
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector", "template")
        self.__copy_template_files(template_dir, self.result_dir)

        # 
        replace = 'labels = ["{}"]'.format('", "'.join(detector.labels))
        with open(self.result_labels_path, "w") as f:
            f.write(replace)
        with open(self.result_boot_py_path) as f:
            boot_py = f.read()
        with open(self.result_boot_py_path, "w") as f:
            target = 'labels = [] # labels'
            boot_py = boot_py.replace(target, replace)
            target = 'anchors = [] # anchors'
            replace = 'anchors = [{}]'.format(', '.join(str(i) for i in detector.anchors))
            boot_py = boot_py.replace(target, replace)
            target = 'sensor.set_windowing((224, 224))'
            replace = 'sensor.set_windowing(({}, {}))'.format(detector.input_shape[1], detector.input_shape[0])
            boot_py = boot_py.replace(target, replace)
            f.write(boot_py)

        return detector, config.detector_result_file_name_prefix

    def __copy_template_files(self, src_dir,  dst_dir):
        files = os.listdir(src_dir)
        for f in files:
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    def zip_dir(self, dir_path, out_zip_file_path):
        shutil.make_archive(os.path.splitext(out_zip_file_path)[0], "zip", dir_path)