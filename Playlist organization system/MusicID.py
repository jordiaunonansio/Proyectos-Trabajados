import cfg
import os.path
import sys
import numpy
import uuid
import eyed3
import vlc
import time
import MusicFiles

class MusicID: 

    __slots__ = ['_UUID_path'] 

    def __init__(self):
        self._UUID_path = {} #diccionari path: uuid

    def __repr__(self):
        string = ''
        for uuid_k, uuid_v in self._UUID_path.items():
            string += str(uuid_k)+' : '+str(uuid_v) + '\n'
        return string

    def __len__(self):
        return len(self._UUID_path)

    def __hash__(self):
        return hash(tuple(self._UUID_path))
    
    def __eq__(self, other):
        return self._UUID_path == other.UUID_path    
    
    def __iter__(self):
        for key in self._UUID_path:
            yield key

    def generate_uuid(self,file): # crea un nou uiid donat un path
        try:
            uuid_create = uuid.uuid5(uuid.NAMESPACE_URL, file)
            if uuid_create  in self._UUID_path.values():
                print("L'arxiu amb path:",file,"No s'utilitzara perque provoca una col·lisió amb la seva representació uuid amb un altre arxiu")
                return None
            else:
                self._UUID_path[file] = uuid_create
                return uuid_create
        except: 
            return ''
    
    def remove_uuid(self, uuid):
        try:
            for key in self._UUID_path.keys():
                if self._UUID_path[key] == uuid:
                    del self._UUID_path[key]
        except:
            pass

    def get_uuid(self, file):
        try:
            return (self._UUID_path[file])
        except:
            return None
        
    def UUID_path(self):
        return self._UUID_path



