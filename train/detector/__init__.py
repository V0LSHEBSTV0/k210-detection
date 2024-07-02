import sys, os
curr_file_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(curr_file_dir)

from utils import gpu_utils, isascii
from utils.logger import Logger
import tempfile
import shutil
import zipfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import re
import kmeans

from train_base import Train_Base


class Detector(Train_Base):
    def __init__(self, input_shape=(224, 224, 3), datasets_dir=None, datasets_zip=None, unpack_dir=None, logger = None,
                max_classes_limit = 15, one_class_min_images_num=100, one_class_max_images_num=2000,
                allow_reshape=False,
                support_shapes=( (224, 224, 3), (240, 240, 3) )
                ):
        '''
            input_shape: input shape (height, width)
            min_images_num: min image number in one class
        '''
        import tensorflow as tf # for multiple process
        self.tf = tf
        self.need_rm_datasets = False
        self.input_shape = input_shape
        self.support_shapes = support_shapes
        if not self.input_shape in self.support_shapes:
            raise Exception("input shape {} not support, only support: {}".format(self.input_shape, self.support_shapes))
        self.allow_reshape = allow_reshape # if dataset image's shape not the same as require's, reshape it
        self.config_max_classes_limit = max_classes_limit
        self.config_one_class_min_images_num = one_class_min_images_num
        self.config_one_class_max_images_num = one_class_max_images_num
        self.datasets_rm_dir = None
        self.model = None
        self.history = None
        self.warning_msg = [] # append warning message here
        if logger:
            self.log = logger

        # unzip datasets
        if datasets_zip:
            self.datasets_dir = self._unpack_datasets(datasets_zip, unpack_dir)
            if not self.datasets_dir:
                self.log.e("can't detect datasets, check zip format")
                raise Exception("can't detect datasets, check zip format")
        elif datasets_dir:
            self.datasets_dir = datasets_dir
        else:
            self.log.e("no datasets args")
            raise Exception("no datasets args")
        

        # load dataset
        ok, msg, self.labels, classes_data_counts, datasets_x, datasets_y = self._load_datasets(self.datasets_dir)
        if not ok:
            msg = f"datasets format error: {msg}"
            self.log.e(msg)
            raise Exception(msg)
        
        # check dataset is valid
        ok, err_msg = self._is_datasets_valid(self.labels, classes_data_counts, one_class_min_images_num=self.config_one_class_min_images_num, one_class_max_images_num=self.config_one_class_max_images_num)
        if not ok:
            self.log.e(err_msg)
            raise Exception(err_msg)
        self.log.i("load datasets complete, check pass, images num:{}, bboxes num:{}".format(len(datasets_x), sum(classes_data_counts)))
        self.datasets_x = np.array(datasets_x, dtype='uint8')
        self.datasets_y = datasets_y

        class _Train_progress_cb(tf.keras.callbacks.Callback):
            '''
                Custom callback class for time of training
            '''
            def __init__(self, epochs, user_progress_callback, logger):
                self.epochs = epochs
                self.logger = logger
                self.user_progress_callback = user_progress_callback

            def on_epoch_begin(self, epoch, logs=None):
                self.logger.i("epoch {} start".format(epoch))

            def on_epoch_end(self, epoch, logs=None):
                self.logger.i("epoch {} end: {}".format(epoch, logs))
                if self.user_progress_callback:
                    self.user_progress_callback((epoch + 1) / self.epochs * 100, "train epoch end")

            def on_train_begin(self, logs=None):
                self.logger.i("train start")
                if self.user_progress_callback:
                    self.user_progress_callback(0, "train start")

            def on_train_end(self, logs=None):
                self.logger.i("train end")
                if self.user_progress_callback:
                    self.user_progress_callback(100, "train end")
        self.Train_progress_cb = _Train_progress_cb

    def __del__(self):
        if self.need_rm_datasets:
            try:
                shutil.rmtree(self.datasets_dir)
                self.log.i(f"clean temp dataset dir:{self.datasets_dir}")
            except Exception as e:
                try:
                    self.log.e("clean temp files error:{}".format(e))
                except Exception:
                    print("log object invalid, var scope usage error, check code")

    def _get_anchors(self, bboxes_in, input_shape=(224, 224), clusters = 5, strip_size = 32):
        '''
            @input_shape tuple (h, w)
            @bboxes_in format: [ [[xmin,ymin, xmax, ymax, label],], ]
                        value range: x [0, w], y [0, h]
            @return anchors, format: 10 value tuple
        '''
        w = input_shape[1]
        h = input_shape[0]
        # TODO: add position to iou, not only box size
        bboxes = []
        for items in bboxes_in:
            for bbox in items:
                bboxes.append(( (bbox[2] - bbox[0])/w, (bbox[3] - bbox[1])/h ))
        bboxes = np.array(bboxes)
        self.log.i(f"bboxes num: {len(bboxes)}, first bbox: {bboxes[0]}")
        out = kmeans.kmeans(bboxes, k=clusters)
        iou = kmeans.avg_iou(bboxes, out) * 100
        self.log.i("bbox accuracy(IOU): {:.2f}%".format(iou))
        self.log.i("bound boxes: {}".format( ",".join("({:f},{:.2f})".format(item[0] * w, item[1] * h) for item in out) ))
        for i, wh in enumerate(out):
            out[i][0] = wh[0]*w/strip_size
            out[i][1] = wh[1]*h/strip_size
        anchors = list(out.flatten())
        self.log.i(f"anchors: {anchors}")
        ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
        self.log.i("w/h ratios: {}".format(sorted(ratios)))
        return anchors

    def train(self, epochs= 100,
                    progress_cb=None,
                    weights=os.path.join(curr_file_dir, "weights", "mobilenet_7_5_224_tf_no_top.h5"),
                    batch_size = 5,
                    train_times = 5,
                    valid_times = 2,
                    learning_rate=1e-4,
                    jitter = False,
                    is_only_detect = False,
                    save_best_weights_path = "out/best_weights.h5",
                    save_final_weights_path = "out/final_weights.h5",
                    ):
        import tensorflow as tf
        from yolo.frontend import create_yolo

        self.log.i("train, labels:{}".format(self.labels))
        self.log.d("train, datasets dir:{}".format(self.datasets_dir))

        # param check
        # TODO: check more param
        if len(self.labels) == 1:
            is_only_detect = True

        self.save_best_weights_path = save_best_weights_path
        self.save_final_weights_path = save_final_weights_path

        # create yolo model
        strip_size = 32 if min(self.input_shape[:2])%32 == 0 else 16
        # get anchors
        self.anchors = self._get_anchors(self.datasets_y, self.input_shape[:2], strip_size = strip_size)
        # create network
        yolo = create_yolo(
                            architecture = "MobileNet",
                            labels = self.labels,
                            input_size = self.input_shape[:2],
                            anchors = self.anchors,
                            coord_scale=1.0,
                            class_scale=1.0,
                            object_scale=5.0,
                            no_object_scale=1.0,
                            weights = weights,
                            strip_size =  strip_size
                )
        self.model = yolo
        # train
        self.history = yolo.train(
                                img_folder = None,
                                ann_folder = None,
                                img_in_mem = self.datasets_x,       # datasets in mem, format: list
                                ann_in_mem = self.datasets_y,       # datasets's annotation in mem, format: list
                                nb_epoch   = epochs,
                                save_best_weights_path = save_best_weights_path,
                                save_final_weights_path = save_final_weights_path,
                                batch_size=batch_size,
                                jitter=jitter,
                                learning_rate=learning_rate, 
                                train_times=train_times,
                                valid_times=valid_times,
                                valid_img_folder="",
                                valid_ann_folder="",
                                valid_img_in_mem = None,
                                valid_ann_in_mem = None,
                                first_trainable_layer=None,
                                is_only_detect = is_only_detect,
                                progress_callbacks = [self.Train_progress_cb(epochs, progress_cb, self.log)]
                        )

    
    def report(self, out_path, limit_y_range=None):
        '''
            generate result charts
        '''
        self.log.i("generate report image")
        if not self.history:
            return
        history = self.history
        # set for server with no Tkagg GUI support, use agg(non-GUI backend)
        plt.switch_backend('agg')
        
        fig, axes = plt.subplots(1, 1, constrained_layout=True, figsize = (16, 10), dpi=100)
        if limit_y_range:
            plt.ylim(limit_y_range)

        # acc and val_acc
        # {'loss': [0.5860330664989357, 0.3398533443955177], 'accuracy': [0.70944744, 0.85026735], 'val_loss': [0.4948340670338699, 0.49342870752194096], 'val_accuracy': [0.7, 0.74285716]}
        if "acc" in history.history:
            kws = {
                "acc": "acc",
                "val_acc": "val_acc",
                "loss": "loss",
                "val_loss": "val_loss"
            }
        else:
            kws = {
                "acc": "accuracy",
                "val_acc": "val_accuracy",
                "loss": "loss",
                "val_loss": "val_loss"
            }

        # axes[0].plot( history.history[kws['acc']], color='#2886EA', label="train")
        # axes[0].plot( history.history[kws['val_acc']], color = '#3FCD6D', label="valid")
        # axes[0].set_title('model accuracy')
        # axes[0].set_ylabel('accuracy')
        # axes[0].set_xlabel('epoch')
        # axes[0].locator_params(integer=True)
        # axes[0].legend()

        # loss and val_loss
        axes.plot( history.history[kws['loss']], color='#2886EA', label="train")
        axes.plot( history.history[kws['val_loss']], color = '#3FCD6D', label="valid")
        axes.set_title('model loss')
        axes.set_ylabel('loss')
        axes.set_xlabel('epoch')
        axes.locator_params(integer=True)
        axes.legend()

        # confusion matrix 
        # import itertools
        
        # cm, labels_idx = self._get_confusion_matrix()
        # axes[2].imshow(cm, interpolation='nearest', cmap = plt.cm.GnBu)
        # axes[2].set_title("confusion matrix")
        # # axes[2].colorbar()
        # num_local = np.array(range(len(labels_idx)))
        # axes[2].set_xticks(num_local)
        # axes[2].set_xticklabels(labels_idx.keys(), rotation=45)
        # axes[2].set_yticks(num_local)
        # axes[2].set_yticklabels(labels_idx.keys())

        # thresh = cm.max() / 2. # front color black or white according to the background color
        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     axes[2].text(j, i, format(cm[i, j], 'd'),
        #             horizontalalignment = 'center',
        #             color = 'white' if cm[i, j] > thresh else "black")
        # axes[2].set_ylabel('True label')
        # axes[2].set_xlabel('Predicted label')


        # save to fs
        fig.savefig(out_path)
        plt.close()
        self.log.i("generate report image end")

    def save(self, h5_path=None, tflite_path=None):
        src_h5_path = self.save_best_weights_path
        if h5_path:
            shutil.copyfile(src_h5_path, h5_path)
        if tflite_path:
            print("save tfilte to :", tflite_path)
            import tensorflow as tf
            model = tf.keras.models.load_model(src_h5_path)
            tf.compat.v1.disable_eager_execution()
            converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(src_h5_path,
                                                output_arrays=['{}/BiasAdd'.format(model.get_layer(None, -2).name)])
            tfmodel = converter.convert()
            with open (tflite_path , "wb") as f:
                f.write(tfmodel)

    def get_sample_images(self, sample_num, copy_to_dir):
        from PIL import Image
        if self.datasets_x is None:
            raise Exception("datasets dir not exists")
        indxes = np.random.choice(range(self.datasets_x.shape[0]), sample_num, replace=False)
        for i in indxes:
            img = self.datasets_x[i]
            path = os.path.join(copy_to_dir, f"image_{i}.jpg")
            img = Image.fromarray(img)
            img.save(path)

    def _get_confusion_matrix(self, batch_size = 5):
        from keras.preprocessing.image import ImageDataGenerator
        from keras.applications.mobilenet import preprocess_input
        valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        valid_data = valid_gen.flow_from_directory(self.datasets_dir,
                target_size=[self.input_shape[0], self.input_shape[1]],
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='sparse',
                shuffle=False
            )
        prediction = self.model.pre()
        predict_labels = np.argmax(prediction, axis=1)
        true_labels = valid_data.classes
        if len(predict_labels) != len(true_labels):
            true_labels = true_labels[0:len(predict_labels)]
        cm = confusion_matrix(true_labels, predict_labels)
        return cm, valid_data.class_indices
        

    def _unpack_datasets(self, datasets_zip, datasets_dir=None, rm_dataset=True):
        '''
            uppack zip datasets to /temp, make /temp as tmpfs is recommend
            zip should be: 
                            datasets
                                   |
                                    ---- tfrecord1
                                   |
                                    ---- tfrecord1
            or: 
                        ---- tfrecord1
                        ---- tfrecord1
        '''
        
        if not datasets_dir:
            datasets_dir = os.path.join(tempfile.gettempdir(), "detector_datasets")
            if rm_dataset:
                self.datasets_rm_dir = datasets_dir
                self.need_rm_datasets = True
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
        zip_file = zipfile.ZipFile(datasets_zip)
        for names in zip_file.namelist():
            zip_file.extract(names, datasets_dir)
        zip_file.close()
        dirs = []
        for d in os.listdir(datasets_dir):
            if d.startswith(".") or not os.path.isdir(os.path.join(datasets_dir, d)):
                continue
            dirs.append(d)
        if len(dirs) == 1: # sub dir
            root_dir = dirs[0]
            datasets_dir = os.path.join(datasets_dir, root_dir)
        elif len(dirs) == 0: # no sub dir
            pass
        else: # multiple folder, not support
            return None
        return datasets_dir

    def _check_update_input_shape(self, img_shape):
        '''
            this will change self.input_shape according to img_shape if suppport
        '''
        if not img_shape in self.support_shapes:
            return False
        self.input_shape = img_shape
        self.log.i(f"input_shape: {self.input_shape}")
        return True

    def _load_datasets(self, datasets_dir):
        '''
            load datasets, support format:
                TFRecord: tfrecord files and tf_label_map.pbtxt in datasets_dir
            @return ok, msg, labels, classes_data_counts, datasets_x, datasets_y
                    classes_data_counts: every class's dataset count, format list, index the same as label's
                    datasets_x: np.ndarray images, not normalize, RGB channel value: [0, 255]
                    datasets_y: np.ndarray bboxes and labels index for one image, format: [[xmin, ymin, xmax, ymax, label_index], ]
                                value range:[0, w] [0, h], not [0, 1]
            @attention self.input_shape can be modified in this function according to the datasets                        
        '''
        def is_pascal_voc():
            dirs = os.listdir(datasets_dir)
            print(datasets_dir)
            if "images" in dirs and "xml" in dirs and "labels.txt" in dirs:
                return True
            return False
        # detect datasets type
        if is_pascal_voc():
            return self._load_datasets_pascal_voc(datasets_dir)
        return False, "datasets error, not support format, please check", [], None, None, None

    def _load_datasets_pascal_voc(self, datasets_dir):
        '''
            load tfrecord, param and return the same as _load_datasets's
        '''
        from parse_pascal_voc_xml import decode_pascal_voc_xml
        from PIL import Image
        labels = []
        datasets_x = []
        datasets_y = []

        img_dir = os.path.join(datasets_dir, "images")
        ann_dir = os.path.join(datasets_dir, "xml")
        labels_path = os.path.join(datasets_dir, "labels.txt")

        # get labels from labels.txt
        labels = []
        with open(labels_path) as f:
            c = f.read()
            labels = c.split()
        # check labels
        ok, msg = self._is_labels_valid(labels)
        if not ok:
            return False, msg, [], None, None, None
        labels_len = len(labels)
        if labels_len < 1:
            return False, 'no classes find', [], None, None, None
        if labels_len > self.config_max_classes_limit:
            return False, 'classes too much, limit:{}, datasets:{}'.format(self.config_max_classes_limit, len(labels)), [], None, None, None
        classes_data_counts = [0] * labels_len
        # get xml path
        xmls = []
        for name in os.listdir(ann_dir):
            print("--", name)
            if name.endswith(".xml"):
                xmls.append(os.path.join(ann_dir, name))
                continue
            if os.path.isdir(os.path.join(ann_dir, name)):
                for sub_name in os.listdir(os.path.join(ann_dir, name)):
                    if sub_name.endswith(".xml"):
                        path = os.path.join(ann_dir, name, sub_name)
                        xmls.append(path)
        # decode xml
        input_shape_checked = False
        for xml_path in xmls:
            ok, result = decode_pascal_voc_xml(xml_path)
            if not ok:
                result = f"decode xml {xml_path} fail, reason: {result}"
                self.on_warning_message(result)
                continue
            # shape
            img_shape = (result['height'], result['width'], result['depth'])
            #  check first image shape, and switch to proper supported input_shape
            if not input_shape_checked:
                if not self._check_update_input_shape(img_shape) and not self.allow_reshape:
                    return False, "not supported input size, supported: {}".format(self.support_shapes), [], None, None, None
                input_shape_checked = True
            if img_shape != self.input_shape:
                msg = f"decode xml {xml_path} ok, but shape {img_shape} not the same as expected: {self.input_shape}"
                if not self.allow_reshape:
                    self.on_warning_message(msg)
                    continue
                else:
                    msg += ", will automatically reshape"
                    self.on_warning_message(msg)
            # load image
            dir_name = os.path.split(os.path.split(result['path'])[0])[-1] # class1 / images
            # images/class1/tututututut.jpg
            img_path = os.path.join(img_dir, dir_name, result['filename'])
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path), dtype='uint8')
            else:
                # images/tututututut.jpg
                img_path = os.path.join(img_dir, result['filename'])
                if os.path.exists(img_path):
                    img = np.array(Image.open(img_path), dtype='uint8')
                else:
                    result = f"decode xml {xml_path}, can not find iamge: {result['path']}"
                    self.on_warning_message(result)
                    continue
            # load bndboxes
            y = []
            for bbox in result['bboxes']:
                if not bbox[4] in labels:
                    result = f"decode xml {xml_path}, can not find iamge: {result['path']}"
                    self.on_warning_message(result)
                    continue
                label_idx = labels.index(bbox[4])
                bbox[4] = label_idx # replace label text with label index
                classes_data_counts[label_idx] += 1
                # range to [0, 1]
                y.append( bbox[:5])
            if len(y) < 1:
                result = f"decode xml {xml_path}, no object, skip"
                self.on_warning_message(result)
                continue
            if img_shape != self.input_shape:
                img, y = self._reshape_image(img, self.input_shape, y)
            datasets_x.append(img)
            datasets_y.append(y)
        return True, "ok", labels, classes_data_counts, datasets_x, datasets_y

    def _decode_pbtxt_file(self, file_path):
        '''
            @return list, if error, will raise Exception
        '''
        res = []
        with open(file_path) as f:
            content = f.read()
            items = re.findall("id: ([0-9].?)\n.*name: '(.*)'", content, re.MULTILINE)
            for i, item in enumerate(items):
                id = int(item[0])
                name = item[1]
                if i != id - 1:
                    raise Exception(f"datasets pbtxt file error, label:{name}'s id should be {i+1}, but now {id}, don't manually edit pbtxt file")
                res.append(name)
        return res
    
    def on_warning_message(self, msg):
        self.log.w(msg)
        self.warning_msg.append(msg)

    def _is_labels_valid(self, labels):
        '''
            labels len should >= 1
            and should be ascii letters, no Chinese or special words
        '''
        if len(labels) < 1:
            err_msg = "labels error: datasets no enough class"
            return False, err_msg
        if len(labels) > self.config_max_classes_limit:
            err_msg = "labels error: too much classes, now {}, but only support {}".format(len(labels), self.config_max_classes_limit)
            return False, err_msg
        for label in labels:
            if not isascii(label):
                return False, "labels error: class name(label) should not contain special letters"
        return True, "ok"

    def _is_datasets_valid(self, labels, classes_dataset_count, one_class_min_images_num=100, one_class_max_images_num=2000):
        '''
            dataset number in every label should > one_class_min_images_num and < one_class_max_images_num
        '''
        for i, label in enumerate(labels):
            # check image number
            if classes_dataset_count[i] < one_class_min_images_num:
                return False, "no enough train images in one class, '{}' only have {}, should > {}, now all datasets num({})".format(label, classes_dataset_count[i], one_class_min_images_num, sum(classes_dataset_count))
            if classes_dataset_count[i] > one_class_max_images_num:
                return False, "too many train images in one class, '{}' have {}, should < {}, now all datasets num({})".format(label, classes_dataset_count[i], one_class_max_images_num, sum(classes_dataset_count))
        return True, "ok"

    def _reshape_image(self, img, to_shape, bboxes):
        raise Exception("not implemented") # TODO: auto reshape images
        new_bboxes = []
        return img, new_bboxes


def train_on_progress(progress, msg):
    print("\n==============")
    print("progress:{}%, msg:{}".format(progress, msg))
    print("==============")

def test_main(datasets_zip, model_path, report_path, log, use_cpu=False):
    import os
    curr_file_dir = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists("out"):
        os.makedirs("out")
    try:
        gpu = gpu_utils.select_gpu(memory_require = 1*1024*1024*1024, tf_gpu_mem_growth=False)
    except Exception:
        gpu = None
    if gpu is None:
        if not use_cpu:
            log.e("no free GPU")
            return 1
        log.i("no GPU, will use [CPU]")
    else:
        log.i("select", gpu)
    detector = Detector(input_shape=(224, 224, 3), datasets_zip=datasets_zip, logger=log, one_class_min_images_num=2)
    detector.train(epochs=2,
                    progress_cb=train_on_progress,
                    weights=os.path.abspath(f"{curr_file_dir}/weights/mobilenet_7_5_224_tf_no_top.h5"),
                    save_best_weights_path = "out/best_weights.h5",
                    save_final_weights_path = "out/final_weights.h5",
                )
    detector.report(report_path)
    detector.save(tflite_path = "out/best_weights.tflite")
    detector.get_sample_images(5, "out/sample_images")
    print("--------result---------")
    print("anchors: {}".format(detector.anchors))
    print("labels:{}".format(detector.labels))
    print("-----------------------")
    if len(detector.warning_msg) > 0:
        print("---------------------")
        print("warining messages:")
        for msg in detector.warning_msg:
            print(msg)
        print("---------------------")

def test():
    log = Logger(file_path="out/train.log")
    if len(sys.argv) >= 4:
        test_main(sys.argv[1], sys.argv[2], sys.argv[3], log, use_cpu=True)
    else:
        import os
        path = os.path.abspath(f"{curr_file_dir}/out")
        path = os.path.join(path, "sample_images")
        if not os.path.exists(path):
            os.makedirs(path)
        test_main(os.path.abspath("../../../../design/assets/test-TFRecords-export.zip"),
                f"{curr_file_dir}/out/classifier.h5",
                f"{curr_file_dir}/out/report.jpg",
                log,
                use_cpu=True)