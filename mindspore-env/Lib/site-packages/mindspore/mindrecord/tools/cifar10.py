# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Cifar10 reader class.
"""
import hashlib
import io
import os
import pickle
import re
import numpy as np

from mindspore import log as logger
from ..core.shardutils import check_filename

__all__ = ['Cifar10']


class CifarMD5Validator:
    '''
    MD5 check for cifar10-batch-py files
    '''
    def __init__(self):
        self.md5_map = {'data_batch_1': 'c99cafc152244af753f735de768cd75f',
                        'data_batch_2': 'd4bba439e000b95fd0a9bffe97cbabec',
                        'data_batch_3': '54ebc095f3ab1f0389bbae665268c751',
                        'data_batch_4': '634d18415352ddfa80567beed471001a',
                        'data_batch_5': '482c414d41f54cd18b22e5b47cb7c3cb',
                        'test_batch': '40351d587109b95175f43aff81a1287e'}

    @staticmethod
    def calculate_md5(file_path):
        """
        Calculate MD5 hash of a file.

        Args:
            file_path (str): Path to the file to calculate MD5 for.

        Returns:
            MD5 hash string if successful, None otherwise.
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
            return file_hash.hexdigest()
        except (IOError, OSError):
            return None

    def check(self, file_path, file_name):
        """
        Check if the file's MD5 matches the expected value.

        Args:
            file_path (str): Path to the file to check.
            file_name (str): Key in md5_map to identify the expected MD5.

        Returns:
            True if MD5 matches, False otherwise (including if file doesn't exist).

        Raises:
            KeyError: If file_name is not found in md5_map.
        """
        expected_md5 = self.md5_map.get(file_name)
        actual_md5 = CifarMD5Validator.calculate_md5(os.path.join(file_path, file_name))

        if actual_md5 is None or expected_md5 is None or actual_md5 != expected_md5:
            logger.warning(f"The MD5 value of {file_name} does not match the official CIFAR10 file."
                           "This file may pose a security risk.")


class RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickle allowing only few safe classes from the builtins module or numpy

    Raises:
        pickle.UnpicklingError: If there is a problem unpickling an object
    """
    def find_class(self, module, name):
        # Only allow safe classes from builtins and numpy
        if module == "numpy.core.multiarray" and name == "_reconstruct":
            return getattr(np.core.multiarray, name)
        if module == "numpy":
            return getattr(np, name)
        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module, name))



def restricted_loads(s):
    """Helper function analogous to pickle.loads()."""
    if isinstance(s, str):
        raise TypeError("can not load pickle from unicode string")
    f = io.BytesIO(s)
    try:
        return RestrictedUnpickler(f, encoding='bytes').load()
    except pickle.UnpicklingError:
        raise RuntimeError("Not a valid Cifar10 Dataset.")
    except UnicodeDecodeError:
        raise RuntimeError("Not a valid Cifar10 Dataset.")
    except Exception:
        raise RuntimeError("Unexpected error while Unpickling Cifar10 Dataset.")


class Cifar10:
    """
    Class to convert cifar10 to MindRecord.

    Args:
        path (str): cifar10 directory which contain data_batch_* and test_batch.
        one_hot (bool): one_hot flag.
    """
    class Test:
        pass

    def __init__(self, path, one_hot=True):
        check_filename(path)
        self.path = path
        if not isinstance(one_hot, bool):
            raise ValueError("The parameter one_hot must be bool")
        self.one_hot = one_hot
        self.images = None
        self.labels = None
        self.validator = CifarMD5Validator()

    def load_data(self):
        """
        Returns a list which contain data & label, test & label.

        Returns:
            list, train images, train labels and test images, test labels
        """
        dic = {}
        images = np.zeros([10000, 3, 32, 32])
        labels = []
        files = os.listdir(self.path)
        for file in files:
            if re.match("data_batch_*", file):
                real_file_path = os.path.realpath(self.path)
                self.validator.check(real_file_path, file)
                with open(os.path.join(real_file_path, file), 'rb') as f: # load train data
                    dic = restricted_loads(f.read())
                    images = np.r_[images, dic[b"data"].reshape([-1, 3, 32, 32])]
                    labels.append(dic[b"labels"])
            elif re.match("test_batch", file):                       # load test data
                real_file_path = os.path.realpath(self.path)
                self.validator.check(real_file_path, file)
                with open(os.path.join(real_file_path, file), 'rb') as f:
                    dic = restricted_loads(f.read())
                    test_images = np.array(dic[b"data"].reshape([-1, 3, 32, 32]))
                    test_labels = np.array(dic[b"labels"])
        dic["train_images"] = images[10000:].transpose(0, 2, 3, 1)
        dic["train_labels"] = np.array(labels).reshape([-1, 1])
        dic["test_images"] = test_images.transpose(0, 2, 3, 1)
        dic["test_labels"] = test_labels.reshape([-1, 1])
        if self.one_hot:
            dic["train_labels"] = self._one_hot(dic["train_labels"], 10)
            dic["test_labels"] = self._one_hot(dic["test_labels"], 10)

        self.images, self.labels = dic["train_images"], dic["train_labels"]
        self.Test.images, self.Test.labels = dic["test_images"], dic["test_labels"]
        return [dic["train_images"], dic["train_labels"], dic["test_images"], dic["test_labels"]]

    def _one_hot(self, labels, num):
        """
        Returns a numpy.

        Returns:
            Object, numpy array.
        """
        size = labels.shape[0]
        label_one_hot = np.zeros([size, num])
        for i in range(size):
            label_one_hot[i, np.squeeze(labels[i])] = 1
        return label_one_hot
