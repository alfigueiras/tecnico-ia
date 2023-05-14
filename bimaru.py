# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

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

from typing import Dict, List, Optional, Tuple
import numpy.typing as npt
import numpy as np


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self):
        parsed_instance = self.parse_instance()
        self.board = parsed_instance["board"]
        self.set_water()

        self.rows: List[int] = parsed_instance["rows"]
        self.columns: List[int] = parsed_instance["columns"]

    def set_water(self):
        for i, r in enumerate(self.rows):
            n_boats = sum([1 for e in self.board[i, :] if e != ""])
            if n_boats == r:
                self.board[i, :][self.board == ""] = "."

        for j, c in enumerate(self.columns):
            n_boats = sum([1 for e in self.board[:, j] if e != ""])
            if n_boats == c:
                self.board[:, j][self.board == ""] = "."

        for m in range(len(self.rows)):
            for n in range(len(self.columns)):
                value = self.board[(m, n)]
                if value not in ["", ".", "W"]:
                    adj_values = self.get_adjacent_values(m, n)
                    if value == "C":
                        for v in adj_values.values():
                            self.set_value(v[0][0], v[0][1], ".")
                    if value == "T":
                        for direction, v in adj_values:
                            if direction != "b":
                                self.set_value(v[0][0], v[0][1], ".")
                    if value == "B":
                        for direction, v in adj_values:
                            if direction != "t":
                                self.set_value(v[0][0], v[0][1], ".")
                    if value == "L":
                        for direction, v in adj_values:
                            if direction != "r":
                                self.set_value(v[0][0], v[0][1], ".")
                    if value == "R":
                        for direction, v in adj_values:
                            if direction != "l":
                                self.set_value(v[0][0], v[0][1], ".")
                    if value == "M":
                        for direction, v in adj_values:
                            if len(direction) > 1:
                                self.set_value(v[0][0], v[0][1], ".")

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[(row, col)]

    def set_value(self, row: int, col: int, value: str):
        self.board[(row, col)] = value

    def get_adjacent_values(self, row: int, col: int) -> Dict[str, Tuple[Tuple[int, int], str]]:
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
            for k, v in adjacent_coords:
                if 't' not in k:
                    res[k] = (v, self.board[v])
        elif row == 9:
            for k, v in adjacent_coords:
                if 'b' not in k:
                    res[k] = (v, self.board[v])
        elif col == 0:
            for k, v in adjacent_coords:
                if 'l' not in k:
                    res[k] = (v, self.board[v])
        elif col == 9:
            for k, v in adjacent_coords:
                if 'r' not in k:
                    res[k] = (v, self.board[v])
        else:
            for k, v in adjacent_coords:
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
            print(line)
            split_line = line.split("\t")
            if split_line[0] == "ROW":
                print(split_line)
                res["rows"]: List[int] = [int(e) for e in split_line[1:]]
            elif split_line[0] == "COLUMN":
                print(split_line)
                res["columns"]: List[int] = [int(e) for e in split_line[1:]]
            elif split_line[0] == "HINT":
                print(split_line)
                board[(int(split_line[1]), int(split_line[2]))] = split_line[-1]
            print(res["rows"])
        res["board"] = board
        print(res)
        return res

    # TODO: outros metodos da classe


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    p = Bimaru(Board())

    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
