from __future__ import print_function, absolute_import
import os.path as osp
import random
import numpy as np
from collections import namedtuple
import os

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json, read_mat
from ..utils.dist_utils import synchronize

def parse_dbStruct(path, time_stamp=True):
    matStruct = read_mat(path)
    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T
    qImage = [f[0].item() for f in matStruct[3+time_stamp]]
    utmQ = matStruct[4+time_stamp].T
    numDb = matStruct[5+time_stamp*2].item()
    numQ = matStruct[6+time_stamp*2].item()

    dbStruct = namedtuple('dbStruct',
            ['dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ'])
    return dbStruct(dbImage, utmDb, qImage, utmQ, numDb, numQ)

class Tokyo(Dataset):

    def __init__(self, root, scale=None, verbose=True):
        super(Tokyo, self).__init__(root)

        self.arrange()
        self.load(verbose)

    def arrange(self):
        if self._check_integrity():
            return

        raw_dir = osp.join(self.root, 'raw')
        if (not osp.isdir(raw_dir)):
            raise RuntimeError("Dataset not found.")
        TM_root = osp.join('tokyoTM', 'images')
        # db_root = osp.join('tokyo247', 'images')

        identities = []
        utms = []
        pids = {}
        pids_ts = {}
        def register_TM(split):
            struct = parse_dbStruct(osp.join(raw_dir, 'tokyoTM_'+split+'.mat'), True)
            struct_images = struct.qImage + struct.dbImage
            struct_utms = np.concatenate((struct.utmQ, struct.utmDb))
            ids = []
            for fpath, utm in zip(struct_images, struct_utms):
                sid = fpath.split('/')[1]
                if (sid not in pids.keys()):
                    pid = len(identities)
                    pids[sid] = pid
                    pids_ts[sid] = []
                    identities.append([])
                    utms.append(utm.tolist())
                    ids.append(pid)
                ts = fpath.split('/')[2]
                if (ts not in pids_ts[sid]):
                    pids_ts[sid].append(ts)
                    identities[pids[sid]].append([])
                id_ts = pids_ts[sid].index(ts)
                if (osp.join(TM_root, fpath) not in identities[pids[sid]][id_ts]):
                    identities[pids[sid]][id_ts].append(osp.join(TM_root, fpath))
                assert(utms[pids[sid]]==utm.tolist())
            return ids

        train_pids = register_TM('train')
        val_pids = register_TM('val')

        new_identities, new_utms = [], []
        new_train_pids = []
        q_val_pids, db_val_pids = [], []
        for p, identity in enumerate(identities):
            if (p in train_pids):
                for sub in identity:
                    new_train_pids.append(len(new_identities))
                    new_identities.append(sorted(sub))
                    new_utms.append(utms[p])
            if (p in val_pids):
                if (len(identity)>1):
                    query_idx = random.randrange(len(identity))
                    query = identity.pop(query_idx)
                    q_val_pids.append(len(new_identities))
                    new_identities.append(sorted(query))
                    new_utms.append(utms[p])
                for sub in identity:
                    db_val_pids.append(len(new_identities))
                    new_identities.append(sorted(sub))
                    new_utms.append(utms[p])

        train_pids = new_train_pids
        identities = new_identities
        utms = new_utms

        # process tokyo247
        raw_dir = osp.join(self.root, 'images')
        # print(raw_dir)
        if (not osp.isdir(raw_dir)):
            raise RuntimeError("Dataset not found.")
        q_pids, db_pids = {}, {}
        def register_247(split):
            dir_path = osp.join(raw_dir, split, 'queries')


            q_ids = []
            if (osp.isdir(dir_path)):
                for img_name in os.listdir(dir_path):
                # for img_name in f:
                    # print(img_name)
                    _, UTM_east, UTM_north, UTM_zone_number, UTM_zone_letter, latitude, longitude, pano_id, tile_num, heading, pitch, roll, height, timestamp, note, extension = img_name.split('@')
                    sid = pano_id
                    utm = [float(UTM_east),float(UTM_north)]
                    if (sid not in q_pids.keys()):      
                        pid = len(identities)
                        q_pids[sid] = pid
                        identities.append([])
                        utms.append(utm)
                        q_ids.append(pid)
                    identities[q_pids[sid]].append(osp.join(dir_path, img_name))
                    assert(utms[q_pids[sid]]==utm)

            dir_path = osp.join(raw_dir, split, 'database')
            db_ids = []
            if (osp.isdir(dir_path)): 
                for img_name in os.listdir(dir_path):
                    # for img_name in f:
                    print(img_name)
                    _, UTM_east, UTM_north, UTM_zone_number, UTM_zone_letter, latitude, longitude, pano_id, tile_num, heading, pitch, roll, height, timestamp, note, extension = img_name.split('@')
                    sid = pano_id
                    utm = [float(UTM_east),float(UTM_north)]
                    # [585089.3603214071, 4477427.575588894]
                    if (sid not in db_pids.keys()):
                        pid = len(identities)
                        db_pids[sid] = pid
                        identities.append([])
                        utms.append(utm)
                        db_ids.append(pid)
                    identities[db_pids[sid]].append(osp.join(dir_path, img_name))
                    assert(utms[db_pids[sid]]==utm)
            return q_ids, db_ids

        q_test_pids, db_test_pids = register_247('test')
        assert len(identities)==len(utms)

        try:
            rank = dist.get_rank()
        except:
            rank = 0

        # Save meta information into a json file
        meta = {'name': 'Tokyo',
                'identities': identities,
                'utm': utms}
        if rank == 0:
            write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the training / test split
        splits = {
            'q_train': sorted(train_pids),
            'db_train': sorted(train_pids),
            'q_val': sorted(q_val_pids),
            'db_val': sorted(db_val_pids),
            'q_test': sorted(q_test_pids),
            'db_test': sorted(db_test_pids)}

        if rank == 0:
            write_json(splits, osp.join(self.root, 'splits.json'))
        synchronize()
