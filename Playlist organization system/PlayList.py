import cfg
import os.path
import sys
import numpy as np
import uuid
import eyed3  #  $ pip install eyed3
import vlc    #  $ pip install python-vlc
import time
from MusicData import MusicData


class PlayList:
    
    __slots__ = ['_playlist', '_obj_music_id', '_audio_player', '_array_playlist']

    def __init__(self, uuid, audioplayer):
        self._playlist = []
        self._obj_music_id = uuid
        self._audio_player = audioplayer
        self._array_playlist = np.array([])
        
    #properties
    @property
    def playlist(self):
        return self._playlist

    @property
    def array_playlist(self):
        return self._array_playlist

    #redefiniciones 
    def __len__(self):
        return len(self._playlist)
    
    def __repr__(self):
        string = ''
        for song in self._playlist:
            string += str(MusicData.get_title(self, song)) + '\n'
        return string

    def __iter__(self):
        for song in self._playlist:
            yield song
    
    def __hash__(self):
        return hash((tuple(self._playlist), self._audio_player, tuple(self._array_playlist)))
    
    def __eq__(self, other):
        return self._playlist == other.playlist() and self.array_playlist == other.array_playlist()
    
    def __ne__(self, other):
        return self._playlist != other.playlist() and self.array_playlist != other.array_playlist()

    #main func
    def load_file(self, file:str):
        self._playlist.clear()
        file_m3u = open(file, 'r', encoding='latin-1')
        for mp3 in file_m3u:
            if mp3[0] != '#' and (mp3[-4:]== '.mp3' or mp3[-5:]=='.mp3\n'):
                if mp3[-5:] == '.mp3\n':
                    mp3 = mp3[:-1]
                repr_uuid = uuid.uuid5(uuid.NAMESPACE_URL, mp3)
                if repr_uuid not in self._playlist:
                    self._playlist.append(repr_uuid)
        self._playlist = list(dict.fromkeys(self._playlist))

    def read_list(self, p_llista):
        self._playlist = list(dict.fromkeys(p_llista))

    def play(self, cmode):
        for uuid in self._playlist:
            self._audio_player.play_song(uuid, cmode)

    #options
    def add_song_at_end(self, uuid:str):
        self._playlist.append(uuid)

    def remove_first_song(self):
        self._playlist = self._playlist[1:]

    def remove_last_song(self):
        self._playlist = self._playlist[:-1]

