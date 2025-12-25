# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
"""TLV format file decoder"""

import struct
from typing import List, Dict, Union

from mindspore import log as logger
from mindspore.profiler.common.constant import FileConstant


class TLVDecoder:
    """
    The class of TLV format data decoder.
    """
    _TYPE_LEN = 2
    _LENGTH_LEN = 4

    @classmethod
    def decode(cls, all_bytes: bytes, fix_data_formats: any, fix_data_struct_size: int) -> List:
        """Decode all the data."""
        result_data = []
        records = cls.tlv_list_decode(all_bytes)
        for record in records:
            if fix_data_struct_size > len(record):
                logger.warning("The collected data has been lost")
                continue
            fix_data_bytes = record[0: fix_data_struct_size]
            tlv_fields = cls.tlv_list_decode(record[fix_data_struct_size:], is_field=True)
            tlv_fields[FileConstant.FIX_SIZE_DATA] = struct.unpack(fix_data_formats, fix_data_bytes)
            result_data.append(tlv_fields)
        return result_data

    @classmethod
    def tlv_list_decode(cls, tlv_bytes: bytes, is_field: bool = False) -> Union[Dict, List]:
        """Decode TLV format data."""
        result_data = {} if is_field else []
        index = 0
        all_bytes_len = len(tlv_bytes)
        while index < all_bytes_len:
            if index + cls._TYPE_LEN > all_bytes_len:
                logger.warning(f"Failed to decode TLV: type field truncated at index {index}")
                break
            type_id = struct.unpack("<H", tlv_bytes[index: index + cls._TYPE_LEN])[0]
            index += cls._TYPE_LEN
            if index + cls._LENGTH_LEN > all_bytes_len:
                logger.warning(f"Failed to decode TLV: length field truncated at index {index}")
                break
            value_len = struct.unpack("<I", tlv_bytes[index: index + cls._LENGTH_LEN])[0]
            index += cls._LENGTH_LEN
            if index + value_len > all_bytes_len:
                logger.warning(f"Failed to decode TLV: value field truncated at index {index}")
                break
            value = tlv_bytes[index: index + value_len]
            index += value_len
            if is_field:
                try:
                    result_data[type_id] = bytes.decode(value)
                except UnicodeDecodeError:
                    logger.warning(f"The collected data can't decode by bytes.decode: {value}")
                    result_data[type_id] = 'N/A'
            else:
                result_data.append(value)
        return result_data
