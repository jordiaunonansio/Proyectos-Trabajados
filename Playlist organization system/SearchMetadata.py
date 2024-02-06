from MusicData import MusicData


class SearchMetadata:
    
    __slots__ = ['_data']

    def __init__(self, data):
        if isinstance(data, MusicData):
            self._data = data
        else:
            raise ValueError()

    def __repr__(self):
        return self._data.__repr__()
    
    def __hash__(self):
        return self._data.__hash__()
    
    def __eq__(self, other):
        return self._data.__eq__(other)

    def __ne__(self, other):
        return self._data.__ne__(other)
    
    #busqueda / recomanació

    def get_similar(self, uuid, max_list) ->list:
        out = {}
        last = uuid
        for song in range(max_list):
            if uuid != last:
                for key, val in self._data.get_next_songs(key):
                    distuk, weightuk = self._data.get_song_distance(last, key)
                    distku, weightku = self._data.get_song_distance(key, last)
                    z1,z2 =0,0
                    if weightuk != 0:
                        z1 = (distuk / weightuk)*(self._data.get_song_rank(last)/2)
                    if weightku !=0:
                        z2 = (distku / weightku)*(self._data.get_song_rank(key)/2) 
                    out[key] = z1+z2
                    last = key
            else:
                for key, val in self._data.get_next_songs(last):
                    distuk, weightuk = self._data.get_song_distance(last, key)
                    distku, weightku = self._data.get_song_distance(key, last)
                    z1,z2 =0,0
                    if weightuk != 0:
                        z1 = (distuk / weightuk)*(self._data.get_song_rank(last)/2)
                    if weightku !=0:
                        z2 = (distku / weightku)*(self._data.get_song_rank(key)/2) 
                    out[key] = z1+z2
                    last = key
        return out
    
    def get_topfive(self):
        nodes = self._data.grafhash_obj.nodes
        out = {}
        for node in nodes:
            out[node] = self._data.get_song_rank(node)
        list = sorted(out)
        list_top_five = list[:5]
        similar = []
        for song in list_top_five:
            for song_aux in self.get_similar(song,5):
                similar.append(song_aux)
        idv_semblanca = {}
        for i in similar:
            idv_semblanca[i] = 0
            rem = []
            for s in similar:
                if i == s:
                    pass
                else:
                    rem.append(s)
            for j in rem:
                idv_semblanca[i] += len(self.get_similar(j,5))
        l_out = sorted(idv_semblanca)
        l_out.reverse()
        return l_out[:5]
            

    #funcions de busqueda
    def title(self, string):
        l= []
        for uuid in self._data._path.keys():
            self._data.load_metadata(uuid)
            if str(self._data.get_title(uuid)).find(str(string).capitalize()) != -1:
                l.append(uuid)
        return l

    def artist(self, string):
        l= []
        for uuid in self._data._path.keys():
            self._data.load_metadata(uuid)
            if str(self._data.get_artist(uuid)).find(str(string).capitalize()) != -1:
                l.append(uuid)
        return l
        
    def album(self, string):
        l= []
        for uuid in self._data._path.keys():
            self._data.load_metadata(uuid)
            if str(self._data.get_album(uuid)).find(str(string).capitalize()) != -1:
                l.append(uuid)
        return l
    
    def genre(self, string):
        l= []
        z = []
        for uuid in self._data._path.keys():
            self._data.load_metadata(uuid)
            for element in self._data.get_genre(uuid):
                z.append(self._data.get_genre(uuid))
                if str(string).capitalize() in element or str(string).capitalize() == element:
                    l.append(uuid)
        return l
    
    #operators
    def or_operator(self, l1, l2):
        l = []
        for element in l1:
            if element not in l:
                l.append(element)

        for element in l2:
            if element not in l:
                l.append(element)
        return l

    def and_operator(self, l1, l2):
        l = []
        for element in l1:
            if element in l2 and element not in l:
                l.append(element)
        for element in l2:
            if element in l1 and element not in l:
                l.append(element)
        return l

