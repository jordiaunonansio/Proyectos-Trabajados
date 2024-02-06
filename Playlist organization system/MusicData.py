import cfg
import os.path
import sys
import numpy
import uuid
import eyed3
import vlc
import time
from MusicID import MusicID 
from ElementData import ElementData
from GrafHash import GrafHash

class MusicData:

    __slots__ = ['_path', '_grafhash_obj']

    def __init__(self):
        self._path = {}
        self._grafhash_obj = GrafHash(digraf=True)

    #funcionalidades redefinidas
    def __len__(self):
        return len(self._path)
    
    def __iter__(self):
        return self._grafhash_obj.__iter__()
    
    def __repr__(self):
        return self._grafhash_obj.__repr__()
    
    def __hash__(self):
        return id(self)
        
    @property
    def grafhash_obj(self):
        return self._grafhash_obj

    #main function
    def load_metadata(self, uuid): #rcarrega i recupera les dades de les cançons
        MI = MusicID()
        dict = MI.UUID_path()
        for key,val in zip(dict.keys(), dict.values()):
            self._path[val] = key
        if uuid is not None and uuid is not '':
            #evitar que es errors amb el genre
            eyed3.log.setLevel("ERROR")
            metadata = eyed3.load(cfg.get_root()+'/'+self._path[uuid]) 
            if metadata is None:
                print("ERROR: Arxiu MP3 erroni!")
                sys.exit(1)
            duration = int(numpy.ceil(metadata.info.time_secs))
            self.add_song(uuid, self._path[uuid])
            title     = metadata.tag.title
            artist    = metadata.tag.artist 
            album     = metadata.tag.album 
            filename = self._path[uuid]
            genre = []
            try: 
                if metadata.tag.genre.name not in genre:
                    genre.append(metadata.tag.genre.name)
                else:
                    genre.append('None')
            except: 
                genre.append('None')
            elem =  ElementData(title = title, artist = artist, album = album, genre = genre, duration = duration, filename = filename)
            self._grafhash_obj.insert_vertex(uuid, elem)
        else:
            print("ERROR: Arxiu MP3 erroni!") 

    def read_playlist(self, obj_playlist: object): # llegeix un objecte playlist i inserta els edges corresponenta al graf
        d = {}
        for i in range(len(obj_playlist.playlist)-1):
            song = obj_playlist.playlist[i]
            next_song = obj_playlist.playlist[i+1]
            if (song, next_song) in d:
                d[(song, next_song)] += 1
            else:
                d[(song, next_song)] = 1
        for key in d:
            self._grafhash_obj.insert_edge(key[0],key[1],d[key])
        
    #implementaciones
    def add_song(self, uuid, file):
        if uuid not in self._path and len(str(uuid))> 8:
            self._path[uuid] = file
            try:
                self.load_metadata(uuid)
            except:
                pass

    def remove_song(self, uuid):
        try:
            self._grafhash_obj.__delitem__(uuid)
        except:
            pass
        if uuid in self._path.keys():
            del self._path[uuid]


    #recomanacions
    def get_song_rank(self, uuid):
        d_out = self._grafhash_obj.edges_out(uuid)
        d_in = self._grafhash_obj.edges_in(uuid)
        rank = 0
        for val in d_out.values():
            rank += val
        for val in d_in.values():
            rank += val
        return rank

    def get_next_songs(self, uuid):
        edges_apuntats = self._grafhash_obj.edges_out(uuid)
        for key, val in edges_apuntats.items():
            yield key, val

    def get_previous_songs(self, uuid):
        edges_apuntadors = self._grafhash_obj.edges_in(uuid)
        for key, val in edges_apuntadors.items():
            yield key, val

    def get_song_distance(self, uuid1, uuid2): # utilitzant les funcions de grafhash troba el cami mes curt i retorna el onombre de nodes i el pes total del cami
        if uuid1 == uuid2:
            return 0,0
        path = self._grafhash_obj.camiMesCurt(uuid1, uuid2) # carrega el cami mes curt
        if len(path) == 0:
            return 0,0
        else:
            wheight = 0
            for uuid in range(len(path)-1):
                if uuid2 in self._grafhash_obj.edges_out(path[uuid]):
                    d = self._grafhash_obj.edges_out(path[uuid])
                    wheight += d[uuid2]
                    return len(path)-1, wheight
                else:
                    d = self._grafhash_obj.edges_out(path[uuid])
                    wheight += d[path[uuid+1]]  # afegeix per a cada edge del cami el pes corresponent
            return 0,0

    #getters
    def get_paths(self):
        return self._path

    def get_title(self, uuid):
        try:
            if self._grafhash_obj.get(uuid).title == '':
                return None
            else:
                self.load_metadata(uuid)
                return self._grafhash_obj.get(uuid).title
        except:
            return None

    def get_artist(self, uuid):
        try:
            if self._grafhash_obj.get(uuid).artist == '':
                return None
            else:    
                self.load_metadata(uuid)
                return self._grafhash_obj.get(uuid).artist
        except: 
            return None

    def get_album(self, uuid):
        try:
            if self._grafhash_obj.get(uuid).album == '':
                return self._grafhash_obj.get(uuid).album
            else:
                self.load_metadata(uuid)
                return self._grafhash_obj.get(uuid).album
        except: 
            return None
    
    def get_genre(self, uuid):
        try:
            if self._grafhash_obj.get(uuid).genre == '':
                self.load_metadata(uuid)
                return self._grafhash_obj.get(uuid).genre
            else:
                self.load_metadata(uuid)
                return self._grafhash_obj.get(uuid).genre
        except: 
            return None

    def get_duration(self, uuid):
        try:
            if self._grafhash_obj.get(uuid).duration == 0:
                self.load_metadata(uuid)
                return self._grafhash_obj.get(uuid).duration
            else:
                self.load_metadata(uuid)
                return self._grafhash_obj.get(uuid).duration
        except:
            return 0

    def get_filename(self, uuid):
        try:
            if self._grafhash_obj.get(uuid).filename == '':
                self.load_metadata(uuid)
                if uuid not in self._path:
                    self._path[uuid] = self._grafhash_obj.get(uuid).filename
                return self._grafhash_obj.get(uuid).filename
            else:
                self.load_metadata(uuid)
                if uuid not in self._path:
                    self._path[uuid] = self._grafhash_obj.get(uuid).filename
                return self._grafhash_obj.get(uuid).filename
        except:
            return None



