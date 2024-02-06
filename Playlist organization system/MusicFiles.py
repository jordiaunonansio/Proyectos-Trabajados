import cfg
import os.path
import sys
import numpy
import uuid
import eyed3
import vlc
import time
import copy 

class MusicFiles:

    __slots__ = ['_files', '_removed', '_added']

    def __init__(self):
        self._files = [] #llista de tots els paths
        self._removed = []
        self._added = []
    
    
    def __len__(self):
        return len(self._files)
        
    def __repr__(self) -> str:
        string = ''
        for file in self._files:
            string = file + ' '
        return string

    def __hash__(self):
        return hash(tuple(self._files))

    def __iter__(self):
        for file in self._files:
            yield file
    
    def __eq__(self, other):
        return self._files == other.files

    def __ne__(self, other):
        return self._files != other.files

    def reload_fs(self, path :str):
        old= self._files
        self._files = []
        #print("Cercant arxius dins [" + cfg.get_root() + "]\n")
        #itera sobre tots els arxius
        for root, dirs, files in os.walk(path):
            for filename in files:
                path_file = (root+'/'+filename) 
                if filename.lower().endswith(tuple(['.mp3'])):
                    self._files.append(path_file)
        self.added(old)
        self.removed(old)

    #calcul de les cançons afegides o eliminades
    def files_added(self):
        return self._added

    def files_removed(self):
        return self._removed

    def added(self, old):
        self._added = []
        for file in self._files:
            if file not in old:
                self._added.append(file)
        
    def removed(self, old):
        self._removed = []
        for file in old:
            if file not in self._files:
                  self._removed.append(file)