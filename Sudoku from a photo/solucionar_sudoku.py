import numpy as np

def troba_posicio(tablero):
    for fila in range(tablero.shape[0]):
        for columna in range(tablero.shape[1]):
            if tablero[fila, columna] == 0:
                return fila, columna
    return None, None
  
def valid(tablero, fila, columna, num):
    for i in range(tablero.shape[1]):
        if tablero[fila, i] == num:
            return False

    for i in range(tablero.shape[0]):
        if tablero[i, columna] == num:
            return False

    capsa_x = columna // 3
    capsa_y = fila // 3

    for i in range(capsa_y * 3, capsa_y * 3 + 3):
        for j in range(capsa_x * 3, capsa_x * 3 + 3):
            if tablero[i, j] == num and (i != fila or j != columna):
                return False
    return True

def main(tablero):
    fila, columna = troba_posicio(tablero)
    
    if fila == None:
        return True
    

    for num in range(1,10):
        correcte = valid(tablero, fila, columna, num)
        
        if correcte:
            tablero[fila, columna] = num
            if main(tablero) == True:
                return True
            tablero[fila, columna] = 0 
    return False


def Sudoku(tablero):
    main(tablero)
    return tablero
    


# Tablero = np.array([
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     [0, 9, 8, 0, 0, 0, 0, 6, 0],
#     [8, 0, 0, 0, 6, 0, 0, 0, 3],
#     [4, 0, 0, 8, 0, 3, 0, 0, 1],
#     [7, 0, 0, 0, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 0, 2, 8, 0],
#     [0, 0, 0, 4, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ])

            

