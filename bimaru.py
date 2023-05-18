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
    # CHANGE LEVEL TO SEE OTHER LOGS
    level="DEBUG",
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, board: npt.ArrayLike[Optional[str]], rows, columns, available_boats={1: 4, 2: 3, 3: 2, 4: 1}, empty_spots_row=np.zeros(10, dtype=int), empty_spots_col=np.zeros(10, dtype=int)):
        self.board = board
        self.rows: List[int] = rows
        self.columns: List[int] = columns
        self.available_boats = available_boats
        self.set_initial_water()
        self.decrease_hint_boats()

        # Tirar?
        self.empty_spots_row = empty_spots_row
        self.empty_spots_col = empty_spots_col

    # Correr isto sempre depois do set_water?
    def find_empty_spots(self):
        for (row, col), value in np.ndenumerate(self.board):
            if value == "":
                self.empty_spots_row[row] += 1
                self.empty_spots_col[col] += 1
        print(self.empty_spots_col)
        print(self.empty_spots_row)


    def set_initial_water(self):
        """Sets water at the first instance of the board."""
        # FILL ALL ROWS THAT ARE KNOWN WITH WATER
        for i, r in enumerate(self.rows):
            n_boats = sum([1 for e in self.board[i, :] if e != "" and e != "." and e != 'W'])
            if n_boats == r:
                for n in range(len(self.columns)):
                    if self.get_value(row=i, col=n) == "":
                        self.set_value(row=i, col=n, value=".")

        # FILL ALL COLUMNS THAT ARE KNOWN WITH WATER
        for j, c in enumerate(self.columns):
            n_boats = sum([1 for e in self.board[:, j] if e != "" and e != "." and e != 'W'])
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
            n_boats = sum([1 for e in self.board[r_i, :] if e != "" and e != "." and e != 'W'])
            logger.debug(n_boats)
            if n_boats == self.rows[r_i]:
                for n in range(len(self.columns)):
                    if self.get_value(row=r_i, col=n) == "":
                        self.set_value(row=r_i, col=n, value=".")

        for c_i in col_indexes:
            n_boats = sum([1 for e in self.board[:, c_i] if e != "" and e != "." and e != 'W'])
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
                            tested_coords.extend([(row + j, col) for j in range(1, k + 2)])
                            self.decrease_available_boats(k + 2)
                            stop = True
                        k += 1
                elif value == "L":
                    while k < 3 and not stop:

                        h_vals = self.adjacent_horizontal_values(row, col + k)
                        if h_vals[1] == "" or h_vals[1] == "W":
                            stop = True
                        elif h_vals[1] == "R":
                            tested_coords.extend([(row, col + j) for j in range(1, k + 2)])
                            self.decrease_available_boats(k + 2)
                            stop = True
                        k += 1

    def set_value(self, row: int, col: int, value: str):
        self.board[(row, col)] = value

    def get_adjacent_values(self, row: int, col: int) -> Dict[str, Tuple[Tuple[int, int], str]]:
        """Returns dictionary with all the adjacent coordinates plus their values."""
        adjacent_coords = {
            't': (row - 1, col),
            'b': (row + 1, col),
            'l': (row, col - 1),
            'r': (row, col + 1),
            'tl': (row - 1, col - 1),
            'tr': (row - 1, col + 1),
            'bl': (row + 1, col - 1),
            'br': (row + 1, col + 1),

        }
        res = {}
        if row == 0:
            for k, v in adjacent_coords.items():
                if 't' not in k:
                    res[k] = (v, self.board[v])
        elif row == 9:
            for k, v in adjacent_coords.items():
                if 'b' not in k:
                    res[k] = (v, self.board[v])
        elif col == 0:
            for k, v in adjacent_coords.items():
                if 'l' not in k:
                    res[k] = (v, self.board[v])
        elif col == 9:
            for k, v in adjacent_coords.items():
                if 'r' not in k:
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
        # possivelmente pode dar problemas porque ocupa muito espaço, se der reduzir o que se passa no estado inicial para apenas a board maybe
        self.initial = BimaruState(boardS)

    # Decidir se se mete as águas junto com os barcos ou só os barcos e depois as águas nas ações
    def actions(self, state: BimaruState):

        # Gerar barcos dependendo da profundidade
        # Começar por fazer pesquisa cega
        # Fazer as grids possíveis de somar
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: BimaruState, action):
        # Somar as grids das ações ao nosso board
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_board = state.boardState.board+action["board"]
        state.boardState.board = new_board
        new_state = BimaruState(Board(
            new_board, action["rows"], action["columns"], state.boardState.available_boats))
        new_state.boardState.decrease_available_boats(action["boat_length"])
        # Isto pode ser lento também, é possível que seja melhor definir as águas na ação e só somar ao invés de correr o set_water sempre
        # Ou então correr set_water apenas para as coordenadas onde foi metido o barco, isso era capaz de ser bastante melhor ideia
        new_state.boardState.set_water()
        new_state.boardState.find_empty_spots()
        return new_state

    def goal_test(self, state: BimaruState):
        # Verificar as regras do goal_test
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        # Dar uma heurística melhor aos barcos maiores e também aos barcos que cumpram as condições das hints
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    parsed = Board.parse_instance()
    board = Board(parsed["board"], parsed["rows"], parsed["columns"])
    board.decrease_hint_boats()
    board.set_water()
    board.find_empty_spots()
    result = depth_first_tree_search(Bimaru(board))

    # result=depth_first_tree_search(prob)
    # print(result)
    p = Bimaru(Board())
    p.board.set_boat(boat_coords=[(2, 9, 't'), (3, 9, 'b')])

    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
