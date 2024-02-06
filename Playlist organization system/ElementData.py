import cfg
import os
import shutil

class ElementData:
    
    __slots__ = ['_title', '_artist', '_album', '_genre', '_duration', '_filename']

    def __init__(self, title = '', artist = '', album = '', genre = '', duration = 0, filename = ''):
        self._title = title
        self._artist = artist
        self._album = album
        self._genre = genre
        self._duration = duration
        self._filename = filename
        
    #redefincio de funcions elementals
    def __hash__(self):
        return hash( self._filename)
    
    def __eq__(self, elem):
        if self._filename == elem._filename:
            return True
        else: return False

    def __len__(self):
        return self._duration

    def __lt__(self, elem):
        if self._filename < elem._filename:
            return True
        else: return False
        
    def __repr__(self) -> str:
        string = ('title: ' + str(self._title)+ ' artist: ' + str(self._artist) +' album: ' +str(self._album)+ ' genre: '+ str(self._genre)+ ' duration: '+ str(self._duration)+'\n')
        return string

    #atributs property (getters)
    @property
    def title (self):
        return self._title

    @property
    def artist(self):
        return self._artist

    @property
    def album(self):
        return self._album

    @property 
    def genre (self):
        return self._genre
    
    @property
    def duration(self):
        return self._duration
    
    @property
    def filename(self):
        return self._filename
 
    #setters
    @title.setter
    def title (self, title):
        self._title = title

    @artist.setter
    def artist(self, artist):
        self._artist = artist

    @album.setter
    def album(self, album):
        self._album = album

    @genre.setter
    def genre (self, genre):
        self._genre = genre
    
    @duration.setter
    def duration(self, duration):
        self._duration =  duration
    
    @filename.setter
    def filename(self, filename):
        self._filename = filename