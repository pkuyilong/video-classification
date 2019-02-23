import subprocess
import os
import pickle

def dump_pkl(store_file, obj):
    try:
        if not os.path.exists(store_file):
            subprocess.call("touch {}".format(store_file), shell=True)
            print("[*] create file success {}!".format(store_file))

        with open(store_file, "wb") as f:
            pickle.dump(obj, f)
        print("store pkl file finish!")

    except Exception as e:
        print(e)



def parse_pkl(store_file):
    try:
        with open(store_file, 'rb') as f:
            d = pickle.load(f)
        return d
    except Exception as e:
        print(e)
