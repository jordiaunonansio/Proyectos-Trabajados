import cfg
import os.path
import sys
import numpy
import uuid
import eyed3  #  $ pip install eyed3
import vlc    #  $ pip install python-vlc
import time
from MusicFiles import MusicFiles
from MusicID import MusicID

class MusicPlayer:

    __slots__ = ['_data']

    def __init__(self, data):
        self._data = data
    
    def __repr__(self):
        return self._data.__repr__()
    
    def __hash__(self):
        return self._data.__hash__()
    
    def __eq__(self, other):
        return self._data.__eq__(other)

    def __ne__(self, other):
        return self._data.__ne__(other)

    def get_data(self):
        return self._data
        
    def print_song(self, uuid: str): #printa les dades de la canço
        if uuid in self._data._path.keys():
            print('\n')
            print('-----------------------------------------------')
            print(" Reproduïnt : "+str(self._data.get_title(uuid)))
            print(" Duració:  "+ str(self._data.get_duration(uuid)))
            print(" Títol:    {}".format(self._data.get_title(uuid)))
            print(" Artista:  {}".format(self._data.get_artist(uuid)))
            print(" Àlbum:{}".format(self._data.get_album(uuid)))
            print(" Gènere:{}".format(self._data.get_genre(uuid)))
            print(" UUID:{}".format(uuid))
            print(" Arxiu:{}".format(self._data.get_paths()[uuid]))
            print('-----------------------------------------------')

    def play_file(self, file: str, mode =0):
        #tret perq no he sigut capaç de que ho passi el caronte
        #player = vlc.MediaPlayer(cfg.get_root()+'/'+file)
        #player.play()'''    
        try:
            metadata = eyed3.load(cfg.get_root()+'/'+file)
        except:
            metadata = None
        if metadata is None: 
            return None
        duration = int(numpy.ceil(metadata.info.time_secs))
        timeout = time.time()+duration
        while True:
            if time.time() < timeout:
                try:
                    time.sleep(1)
                except KeyboardInterrupt: # STOP amb <CTRL>+<C> a la consola
                    break 
            else:
                break
            

    def play_song(self,uuid :str, mode: int): # depenent  del mode, només printa o reprodueix o ambdues, la canço amb determinat uuid
        if mode == 0:
            self.print_song(uuid)
        elif mode == 1:
            self.print_song(uuid)
            self.play_file(self._data.get_paths()[uuid])
        elif mode == 2:
            self.play_file(self._data.get_paths()[uuid])


