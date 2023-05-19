# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2
import logging
import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)
import os
from typing import Dict, List, Optional, Tuple
import numpy.typing as npt
import numpy as np

logging.basicConfig(
    # CHANGE LEVEL TO SEE ONLY INFO LOGS
    level="DEBUG",
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(
        self,
        board,
        rows,
        columns,
        available_boats=None,
    ):
        # ARGUMENTO DEFAULT COMO DICIONÁRIO ESTAVA A DAR UM WARNING
        if available_boats is None:
            available_boats = {1: 4, 2: 3, 3: 2, 4: 1}

        self.board = board
        self.rows: List[int] = rows
        self.columns: List[int] = columns
        self.available_boats = available_boats
        self.decrease_hint_boats()

        self.empty_spots_row = np.zeros(10, dtype=int)
        self.empty_spots_col = np.zeros(10, dtype=int)
        self.find_empty_spots()

    # Correr isto sempre depois do set_water?
    def find_empty_spots(self):
        for (row, col), value in np.ndenumerate(self.board):
            if value == "":
                self.empty_spots_row[row] += 1
                self.empty_spots_col[col] += 1

    def horizontal_boats(self, row, boat_length):
        board_row = self.board[row, :]
        j = 0
        boat_coords = []
        if boat_length == 1:
            for i, val in enumerate(board_row):
                if val == "":
                    coord = (row, i)
                    if self.empty_spots_col[i] >= 1:
                        can_put = True
                        for value in self.get_adjacent_values(row, i).values():
                            if value[1] not in ["W", ".", ""]:
                                can_put = False
                                break
                        if can_put:
                            boat_coords.append([coord])
        if boat_length == 2:
            j = 0
            for i, val in enumerate(board_row):
                if val == "L":
                    j = 1
                elif val == "" or val == "R":
                    j += 1
                    if j >= 2:
                        char_b = ["l", "r"]
                        coords = [
                            (row, i + k - 1, char_b[k])
                            for k in range(0, 2)
                            if char_b[k].upper() != self.get_value(row, i - k - 1)
                        ]
                        can_put = True
                        for cor in coords:
                            if self.empty_spots_col[cor[1]] >= 1:

                                for key, value in self.get_adjacent_values(
                                    cor[0], cor[1]
                                ).items():
                                    if value[1] not in ["W", ".", ""]:

                                        if (
                                            key == "l"
                                            and value[1] == "L"
                                            and cor[2] == "r"
                                        ):                                          
                                            pass
                                        if (                                            
                                            key == "r"
                                            and value[1] == "R"
                                            and cor[2] == "l"
                                        ):                        
                                            pass
                                        else:
                                            can_put = False
                                            break
                            else:
                                break
                        if can_put:
                            boat_coords.append(coords)
                    if val == "R":
                        j = 0
                else:
                    j = 0
        if boat_length == 3:
            j = 0
            for i, val in enumerate(board_row):
                if val == "L":
                    j = 1
                elif val == "M":
                    j += 1
                elif val == "" or val == "R":
                    j += 1
                    if j >= 3:
                        char_b = ["l", "m", "r"]
                        coords = [
                            (row, i + k - 2, char_b[k])
                            for k in range(0, 3)
                            if char_b[k].upper() != self.get_value(row, i + k - 2)
                        ]
                        # Ter cuidado com isto
                        if self.get_value(row,i - 2) == "M":
                            coords = []
                            j = 1
                        can_put = True                        
                        for cor in coords:
                            if self.empty_spots_col[cor[1]] >= 1:
                                for key, value in self.get_adjacent_values(
                                    cor[0], cor[1]
                                ).items():
                                    if value[1] not in ["W", ".", ""]:
                                        if (
                                            key == "l"
                                            and value[1] == "L"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "r"
                                            and value[1] == "R"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "r"
                                            and value[1] == "M"
                                            and cor[2] == "l"
                                        ):
                                            pass
                                        elif (
                                            key == "l"
                                            and value[1] == "M"
                                            and cor[2] == "r"
                                        ):
                                            pass
                                        else:
                                            can_put = False
                                            break
                            else:
                                break
                        if can_put and coords!=[]:
                            boat_coords.append(coords)
                    if val == "R":
                        j = 0
                else:
                    j = 0
        # Possivelmente posso adaptar o de 3 mas it is what it is
        if boat_length == 4:
            j = 0
            for i, val in enumerate(board_row):
                if val == "L":
                    j = 1
                elif val == "M":
                    j += 1
                elif val == "" or val == "R":
                    j += 1
                    if j >= 4:
                        char_b = ["l", "m", "m", "r"]
                        coords = [
                            (row, i + k - 3, char_b[k])
                            for k in range(0, 4)
                            if char_b[k].upper() != self.get_value(row, i + k - 3)
                        ]
                        # cuidado
                        if self.get_value(row,i - 3) == "M":
                            coords = []
                            j = 2
                        can_put = True
                        for cor in coords:
                            if self.empty_spots_col[cor[1]] >= 1:
                                for key, value in self.get_adjacent_values(
                                    cor[0], cor[1]
                                ).items():
                                    if value[1] not in ["W", ".", ""]:
                                        if (
                                            key == "l"
                                            and value[1] == "L"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "r"
                                            and value[1] == "R"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "r"
                                            and value[1] == "M"
                                            and cor[2] == "l"
                                        ):
                                            pass
                                        elif (
                                            key == "l"
                                            and value[1] == "M"
                                            and cor[2] == "r"
                                        ):
                                            pass
                                        elif (
                                            key == "l"
                                            and value[1] == "M"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "r"
                                            and value[1] == "M"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        else:
                                            can_put = False
                                            break
                            else:
                                break
                        if can_put and coords!=[]:
                            boat_coords.append(coords)
                    if val == "R":
                        #Ou zero? Acho que assim é melhor porque nunca pode estar um barco a começar na posição seguinte
                        j = -1
                else:
                    j = 0
        return boat_coords

    def vertical_boats(self, col, boat_length):
        board_col = self.board[:, col]
        j = 0
        boat_coords = []
        if boat_length == 1:
            for i, val in enumerate(board_col):
                if val == "":
                    coord = (i,col)
                    if self.empty_spots_row[i] >= 1:
                        can_put = True
                        for value in self.get_adjacent_values(i,col).values():
                            if value[1] not in ["W", ".", ""]:
                                can_put = False
                                break
                        if can_put:
                            boat_coords.append([coord])
        if boat_length == 2:
            j = 0
            for i, val in enumerate(board_col):
                if val == "T":
                    j = 1
                elif val == "" or val == "B":
                    j += 1
                    if j >= 2:
                        char_b = ["t", "b"]
                        coords = [
                            (i + k - 1, col, char_b[k])
                            for k in range(0, 2)
                            if char_b[k].upper() != self.get_value(i + k - 1,col)
                        ]
                        can_put = True
                        for cor in coords:
                            if self.empty_spots_row[cor[0]] >= 1:
                                for key, value in self.get_adjacent_values(
                                    cor[0], cor[1]
                                ).items():
                                    if value[1] not in ["W", ".", ""]:
                                        if (
                                            key == "t"
                                            and value[1] == "T"
                                            and cor[2] == "b"
                                        ):                                          
                                            pass
                                        if (                                            
                                            key == "b"
                                            and value[1] == "B"
                                            and cor[2] == "t"
                                        ):                        
                                            pass
                                        else:
                                            can_put = False
                                            break
                            else:
                                break
                        if can_put:
                            boat_coords.append(coords)
                    if val == "B":
                        j = -1
                else:
                    j = 0
        if boat_length == 3:
            j = 0
            for i, val in enumerate(board_col):
                if val == "T":
                    j = 1
                elif val == "M":
                    j += 1
                elif val == "" or val == "B":
                    j += 1
                    if j >= 3:
                        char_b = ["t", "m", "b"]
                        coords = [
                            (i + k - 2, col, char_b[k])
                            for k in range(0, 3)
                            if char_b[k].upper() != self.get_value(i + k - 2, col)
                        ]
                        # Ter cuidado com isto
                        if self.get_value(i - 2, col) == "M":
                            coords = []
                            j = 1
                        can_put = True                        
                        for cor in coords:
                            if self.empty_spots_row[cor[0]] >= 1:
                                for key, value in self.get_adjacent_values(
                                    cor[0], cor[1]
                                ).items():
                                    if value[1] not in ["W", ".", ""]:
                                        if (
                                            key == "t"
                                            and value[1] == "T"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "b"
                                            and value[1] == "B"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "b"
                                            and value[1] == "M"
                                            and cor[2] == "t"
                                        ):
                                            pass
                                        elif (
                                            key == "t"
                                            and value[1] == "M"
                                            and cor[2] == "b"
                                        ):
                                            pass
                                        else:
                                            can_put = False
                                            break
                            else:
                                break
                        if can_put and coords!=[]:
                            boat_coords.append(coords)
                    if val == "B":
                        j = -1
                else:
                    j = 0
        if boat_length == 4:
            j = 0
            for i, val in enumerate(board_col):
                if val == "T":
                    j = 1
                elif val == "M":
                    j += 1
                elif val == "" or val == "B":
                    j += 1
                    if j >= 4:
                        char_b = ["t", "m", "m", "b"]
                        coords = [
                            (i + k - 3, col, char_b[k])
                            for k in range(0, 4)
                            if char_b[k].upper() != self.get_value(i + k - 3, col)
                        ]
                        # cuidado
                        if self.get_value(i - 3, col) == "M":
                            coords = []
                            j = 2
                        can_put = True
                        for cor in coords:
                            if self.empty_spots_row[cor[0]] >= 1:
                                for key, value in self.get_adjacent_values(
                                    cor[0], cor[1]
                                ).items():
                                    if value[1] not in ["W", ".", ""]:
                                        if (
                                            key == "t"
                                            and value[1] == "T"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "b"
                                            and value[1] == "B"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "b"
                                            and value[1] == "M"
                                            and cor[2] == "t"
                                        ):
                                            pass
                                        elif (
                                            key == "t"
                                            and value[1] == "M"
                                            and cor[2] == "b"
                                        ):
                                            pass
                                        elif (
                                            key == "t"
                                            and value[1] == "M"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        elif (
                                            key == "b"
                                            and value[1] == "M"
                                            and cor[2] == "m"
                                        ):
                                            pass
                                        else:
                                            can_put = False
                                            break
                            else:
                                break
                        if can_put and coords!=[]:
                            boat_coords.append(coords)
                    if val == "B":
                        j = -1
                else:
                    j = 0
        return boat_coords

    def set_initial_water(self):
        """Sets water at the first instance of the board."""
        # FILL ALL ROWS THAT ARE KNOWN WITH WATER
        for i, r in enumerate(self.rows):
            n_boats = sum(
                [1 for e in self.board[i, :] if e != "" and e != "." and e != "W"]
            )
            if n_boats == r:
                for n in range(len(self.columns)):
                    if self.get_value(row=i, col=n) == "":
                        self.set_value(row=i, col=n, value=".")

        # FILL ALL COLUMNS THAT ARE KNOWN WITH WATER
        for j, c in enumerate(self.columns):
            n_boats = sum(
                [1 for e in self.board[:, j] if e != "" and e != "." and e != "W"]
            )
            if n_boats == c:
                for m in range(len(self.rows)):
                    if self.get_value(row=m, col=j) == "":
                        self.set_value(row=m, col=j, value=".")

        for (m, n), value in np.ndenumerate(self.board):
            if value not in ["", ".", "W"]:
                adj_values = self.get_adjacent_values(m, n)
                if value == "C":
                    for v in adj_values.values():
                        self.set_value(v[0][0], v[0][1], ".")
                if value == "T":
                    for direction, v in adj_values.items():
                        if direction != "b":
                            self.set_value(v[0][0], v[0][1], ".")
                if value == "B":
                    for direction, v in adj_values.items():
                        if direction != "t":
                            self.set_value(v[0][0], v[0][1], ".")
                if value == "L":
                    for direction, v in adj_values.items():
                        if direction != "r":
                            self.set_value(v[0][0], v[0][1], ".")
                if value == "R":
                    for direction, v in adj_values.items():
                        if direction != "l":
                            self.set_value(v[0][0], v[0][1], ".")
                if value == "M":
                    for direction, v in adj_values.items():
                        if len(direction) > 1:
                            self.set_value(v[0][0], v[0][1], ".")

        logger.info(self.board)
        logger.info(f"ROWS: {self.rows}")
        logger.info(f"COLUMNS: {self.columns}")

    def set_boat(self, boat_coords: List[Tuple[int, int, str]]):
        """Set water around placed boat."""
        # SETTING BOAT
        for boat_piece in boat_coords:
            self.set_value(row=boat_piece[0], col=boat_piece[1], value=boat_piece[2])

        # RESOLVING WATER
        boat_coords = [(e[0], e[1]) for e in boat_coords]
        boat_adj = []
        for coord in boat_coords:
            adj_coords = self.get_adjacent_values(coord[0], coord[1])
            adj_coords = [v[0] for v in adj_coords.values()]
            boat_adj.extend(adj_coords)

        boat_adj = set(boat_adj) - set(boat_coords)
        for coord in boat_adj:
            self.set_value(row=coord[0], col=coord[1], value=".")

        # THE PART BELOW SHOULD BE A SEPARATE FUNCTION
        # IT IS REPEATED
        row_indexes = [e[0] for e in boat_coords]
        col_indexes = [e[1] for e in boat_coords]
        for r_i in row_indexes:
            n_boats = sum(
                [1 for e in self.board[r_i, :] if e != "" and e != "." and e != "W"]
            )
            logger.debug(n_boats)
            if n_boats == self.rows[r_i]:
                for n in range(len(self.columns)):
                    if self.get_value(row=r_i, col=n) == "":
                        self.set_value(row=r_i, col=n, value=".")

        for c_i in col_indexes:
            n_boats = sum(
                [1 for e in self.board[:, c_i] if e != "" and e != "." and e != "W"]
            )
            if n_boats == self.columns[c_i]:
                for m in range(len(self.rows)):
                    if self.get_value(row=m, col=c_i) == "":
                        self.set_value(row=m, col=c_i, value=".")

        logger.info(self.board)
        logger.info(f"ROWS: {self.rows}")
        logger.info(f"COLUMNS: {self.columns}")

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[(row, col)]

    def decrease_available_boats(self, boat_length):
        """Tira da lista de barcos disponíveis
        um barco de tamanho pretendido"""
        self.available_boats[boat_length] -= 1

    def decrease_hint_boats(self):
        """Deteta os barcos que já foram colocados e
        subtrai-os aos barcos disponíveis"""
        tested_coords = []
        for (row, col), value in np.ndenumerate(self.board):
            if (row, col) not in tested_coords:
                tested_coords.append((row, col))
                k = 0
                stop = False
                if value == "C":
                    self.decrease_available_boats(1)
                elif value == "T":
                    while k < 3 and not stop:
                        v_vals = self.adjacent_vertical_values(row + k, col)
                        if v_vals[1] == "" or v_vals[1] == "W":
                            stop = True
                        elif v_vals[1] == "B":
                            tested_coords.extend(
                                [(row + j, col) for j in range(1, k + 2)]
                            )
                            self.decrease_available_boats(k + 2)
                            stop = True
                        k += 1
                elif value == "L":
                    while k < 3 and not stop:
                        h_vals = self.adjacent_horizontal_values(row, col + k)
                        if h_vals[1] == "" or h_vals[1] == "W":
                            stop = True
                        elif h_vals[1] == "R":
                            tested_coords.extend(
                                [(row, col + j) for j in range(1, k + 2)]
                            )
                            self.decrease_available_boats(k + 2)
                            stop = True
                        k += 1

    def set_value(self, row: int, col: int, value: str):
        self.board[(row, col)] = value

    def get_adjacent_values(
        self, row: int, col: int
    ) -> Dict[str, Tuple[Tuple[int, int], str]]:
        """Returns dictionary with all the adjacent coordinates plus their values."""
        adjacent_coords = {
            "t": (row - 1, col),
            "b": (row + 1, col),
            "l": (row, col - 1),
            "r": (row, col + 1),
            "tl": (row - 1, col - 1),
            "tr": (row - 1, col + 1),
            "bl": (row + 1, col - 1),
            "br": (row + 1, col + 1),
        }
        res = {}
        if row == 0:
            for k, v in adjacent_coords.items():
                # if t != k?
                if "t" not in k:
                    res[k] = (v, self.board[v])
        elif row == 9:
            for k, v in adjacent_coords.items():
                if "b" not in k:
                    res[k] = (v, self.board[v])
        elif col == 0:
            for k, v in adjacent_coords.items():
                if "l" not in k:
                    res[k] = (v, self.board[v])
        elif col == 9:
            for k, v in adjacent_coords.items():
                if "r" not in k:
                    res[k] = (v, self.board[v])
        else:
            for k, v in adjacent_coords.items():
                res[k] = (v, self.board[v])

        return res

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        if row == 0:
            # TOP OF THE BOARD
            res = (None, self.board[(row + 1, col)])
        elif row == 9:
            # BOTTOM OF THE BOARD
            res = (self.board[(row - 1, col)], None)
        else:
            res = (self.board[(row - 1, col)], self.board[(row + 1, col)])
        return res

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if col == 0:
            # LEFT OF THE BOARD
            res = (None, self.board[(row, col + 1)])
        elif row == 9:
            # RIGHT OF THE BOARD
            res = (self.board[(row, col - 1)], None)
        else:
            res = (self.board[(row, col - 1)], self.board[(row, col + 1)])

        return res

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        res = {}
        board: npt.ArrayLike[Optional[str]] = np.empty([10, 10], dtype=str)
        board[:] = ""
        for line in sys.stdin:
            split_line = line.split("\t")
            if split_line[0] == "ROW":
                # Mudar para np arrays?
                res["rows"]: List[int] = [int(e) for e in split_line[1:]]
            elif split_line[0] == "COLUMN":
                res["columns"]: List[int] = [int(e) for e in split_line[1:]]
            elif split_line[0] == "HINT":
                board[(int(split_line[1]), int(split_line[2]))] = split_line[-1]
        res["board"] = board
        return res


class BimaruState:
    state_id: int = 0

    def __init__(self, board: Board):
        self.boardState = board
        self.depht = 0
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Bimaru(Problem):
    def __init__(self, boardS: Board):
        """O construtor especifica o estado inicial."""
        # possivelmente pode dar problemas porque ocupa muito espaço, se der reduzir o que se passa no estado inicial
        # para apenas a board maybe
        self.initial = BimaruState(boardS)

    # Decidir se se mete as águas junto com os barcos ou só os barcos e depois as águas nas ações
    def actions(self, state: BimaruState):
        # talvez gerar também todos os estados possíveis que se podem ligar às hints existentes em vez de tamanho fixo
        # Gerar barcos dependendo da profundidade
        # Começar por fazer pesquisa cega
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        boat_length = 0
        for key in range(1, 5):
            if state.boardState.available_boats[key] != 0:
                boat_length = key

        for i in range(len(state.boardState.empty_spots_row)):
            if state.boardState.empty_spots_row[i]>=boat_length:
                boats=state.boardState.horizontal_boats(i,boat_length)
                for boat in boats:
                    actions.append({"coords": boat, "boat_length": boat_length})

        for j in range(len(state.boardState.empty_spots_col)):
            if state.boardState.empty_spots_col[j]>=boat_length:
                boats=state.boardState.vertical_boats(i,boat_length)
                for boat in boats:
                    actions.append({"coords": boat, "boat_length": boat_length})
        return actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_state = BimaruState(
            Board(
                state.boardState.board,
                state.boardState.rows,
                state.boardState.columns,
                state.boardState.available_boats,
            )
        )

        new_state.boardState.decrease_available_boats(action["boat_length"])
        new_state.boardState.set_boat(action["coords"])
        new_state.boardState.find_empty_spots()
        return new_state

    def goal_test(self, state: BimaruState):
        # Verificar as regras do goal_test
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        solved = True

        curr_board = state.boardState.board
        rows = state.boardState.rows
        cols = state.boardState.columns

        for r_i in range(len(rows)):
            n_boats = sum(
                [1 for e in curr_board[r_i, :] if e != "" and e != "." and e != "W"]
            )
            if n_boats != rows[r_i]:
                solved = False
                return solved

        for c_i in range(len(cols)):
            n_boats = sum(
                [1 for e in curr_board[:, c_i] if e != "" and e != "." and e != "W"]
            )
            if n_boats != cols[c_i]:
                solved = False
                return solved

        for (r, c), value in np.ndenumerate(curr_board):
            if curr_board[r, c] == "":
                solved = False
                return solved
        return solved


    def h(self, node: Node):
        # Dar uma heurística melhor aos barcos maiores e também aos barcos que cumpram as condições das hints
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    parsed = Board.parse_instance()
    initial_board = Board(parsed["board"], parsed["rows"], parsed["columns"])
    initial_board.decrease_hint_boats()
    initial_board.set_initial_water()
    initial_board.find_empty_spots()
    problem=Bimaru(initial_board)
    result = depth_first_tree_search(problem)
    logger.info(result)
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
