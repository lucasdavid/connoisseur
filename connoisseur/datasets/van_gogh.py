"""VanGogh Dataset.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

from .base import DataSet


class VanGogh(DataSet):
    SOURCE = 'https://ndownloader.figshare.com/files/5870145'
    EXPECTED_SIZE = 5707509034
    EXTRACTED_FOLDER = 'vgdb_2016'
