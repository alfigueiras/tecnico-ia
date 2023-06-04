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

import numpy as np


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(
            self,
            board,
            rows,
            columns,
            hints,
            available_boats=[4, 3, 2, 1]
    ):
        # ARGUMENTO DEFAULT COMO DICIONÁRIO ESTAVA A DAR UM WARNING

        self.board = board
        self.rows = rows
        self.columns = columns
        self.available_boats = available_boats
        self.hints = hints
        self.empty_spots_row = np.zeros(10, dtype=int)
        self.empty_spots_col = np.zeros(10, dtype=int)
        self.boats_placed_row = np.zeros(10, dtype=int)
        self.boats_placed_col = np.zeros(10, dtype=int)
        self.boats_hint_row = np.zeros(10, dtype=int)
        self.boats_hint_col = np.zeros(10, dtype=int)

    def find_boats_and_empty_spots(self):
        """Conta o número de espaços vazios, os barcos colocados, e o número de hints por linha e coluna"""
        for (row, col), value in np.ndenumerate(self.board):
            if value == "":
                self.empty_spots_row[row] += 1
                self.empty_spots_col[col] += 1
            elif value not in [".", "", "W"]:
                self.boats_placed_row[row] += 1
                self.boats_placed_col[col] += 1
                if value.isupper():
                    self.boats_hint_row[row] += 1
                    self.boats_hint_col[col] += 1

    def horizontal_boats(self, row, boat_length):
        """Gera todas as posições possíveis para um barco horizontal de comprimento boat_length na linha escolhida"""
        board_row = self.board[row, :]
        j = 0
        char_lr = [["l", "r"], ["l", "m", "r"], ["l", "m", "m", "r"]]
        boat_coords = []
        coords = None
        for col, val in np.ndenumerate(board_row):
            # COL VEM COMO UM TUPLO DE COMPRIMENTO 1
            col = col[0]
            if val == "":
                j += 1
                if boat_length == 1:
                    coords = [(row, col, "c")]
                else:
                    if j >= boat_length:
                        coords = [(row, col + k - boat_length, char_lr[boat_length - 2][k - 1]) for k in
                                  range(1, boat_length + 1)]
                can_put = True
                if coords:
                    for cor in coords:
                        if self.columns[cor[1]] - self.boats_placed_col[cor[1]] >= 1:
                            for key, value in self.get_adjacent_values(row, col).items():
                                if value[1] not in ["W", ".", ""]:
                                    can_put = False
                                    break
                        else:
                            can_put = False
                            break
                    if can_put:
                        boat_coords.append(coords)
            elif val == "." or val == "W":
                j = 0
            else:
                j = -1
        return boat_coords

    def vertical_boats(self, col, boat_length):
        """Gera todas as posições possíveis para um barco vertical de comprimento boat_length na coluna escolhida"""
        board_col = self.board[:, col]
        j = 0
        char_tb = [["t", "b"], ["t", "m", "b"], ["t", "m", "m", "b"]]
        boat_coords = []
        coords = None
        for row, val in np.ndenumerate(board_col):
            # ROW VEM COMO UM TUPLO DE COMPRIMENTO 1
            row = row[0]
            if val == "":
                j += 1
                if boat_length == 1:
                    coords = [(row, col, "c")]
                else:
                    if j >= boat_length:
                        coords = [(row + k - boat_length, col, char_tb[boat_length - 2][k - 1]) for k in
                                  range(1, boat_length + 1)]
                can_put = True
                if coords:
                    for cor in coords:
                        if self.rows[cor[0]] - self.boats_placed_row[cor[0]] >= 1:
                            for key, value in self.get_adjacent_values(row, col).items():
                                if value[1] not in ["W", ".", ""]:
                                    can_put = False
                                    break
                        else:
                            can_put = False
                            break
                    if can_put:
                        boat_coords.append(coords)
            elif val == "." or val == "W":
                j = 0
            else:
                j = -1
        return boat_coords

    def set_initial_water(self):
        """Coloca as águas conhecidas à partida no estado inicial."""
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

    def set_boat(self, boat_coords):
        """Coloca o barco nas coordernadas dadas e todas as águas possíveis à volta do barco.
        Se uma linha ou coluna ficar com o número total desejado de peças preenchidas, preenche toda essa linha e/ou coluna
        de águas"""
        new_board = np.copy(self.board)

        # SETTING BOAT
        for boat_piece in boat_coords:
            new_board[(boat_piece[0], boat_piece[1])] = boat_piece[2]

        # RESOLVING WATER
        boat_coords = [(e[0], e[1]) for e in boat_coords]
        boat_adj = []
        for coord in boat_coords:
            adj_coords = self.get_adjacent_values(coord[0], coord[1])
            adj_coords = [v[0] for v in adj_coords.values()]
            boat_adj.extend(adj_coords)

        boat_adj = set(boat_adj) - set(boat_coords)
        for coord in boat_adj:
            if self.get_value(coord[0], coord[1]) != "W":
                new_board[(coord[0], coord[1])] = "."

        # THE PART BELOW SHOULD BE A SEPARATE FUNCTION
        # IT IS REPEATED
        row_indexes = [e[0] for e in boat_coords]
        col_indexes = [e[1] for e in boat_coords]
        for r_i in row_indexes:
            n_boats = sum(
                [1 for e in new_board[r_i, :] if e != "" and e != "." and e != "W"]
            )
            if n_boats == self.rows[r_i]:
                for n in range(len(self.columns)):
                    if new_board[(r_i, n)] == "":
                        new_board[(r_i, n)] = "."

        for c_i in col_indexes:
            n_boats = sum(
                [1 for e in new_board[:, c_i] if e != "" and e != "." and e != "W"]
            )
            if n_boats == self.columns[c_i]:
                for m in range(len(self.rows)):
                    if new_board[(m, c_i)] == "":
                        new_board[(m, c_i)] = "."

        return new_board

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[(row, col)]

    def decrease_available_boats(self, boat_length):
        """Tira da lista de barcos disponíveis
        um barco de tamanho pretendido"""
        return [self.available_boats[i] if i != boat_length - 1 else self.available_boats[i] - 1 for i in range(4)]

    def generate_hint_boats(self, row, col, val, max_length=4):
        """Dadas as coordenadas de uma hint, gera todos os barcos possíveis que encaixam nessa hint
        com comprimento máximo igual ao max_length"""
        boat_coords = []
        tb = [["t", "b"], ["t", "m", "b"], ["t", "m", "m", "b"]]
        lr = [["l", "r"], ["l", "m", "r"], ["l", "m", "m", "r"]]
        if val == "M":
            res = []
            if row - 1 >= 0:
                res += self.generate_hint_boats(row - 1, col, "T", max_length)
            if row + 1 < 10:
                res += self.generate_hint_boats(row + 1, col, "B", max_length)
            if col - 1 >= 0:
                res += self.generate_hint_boats(row, col - 1, "L", max_length)
            if col + 1 < 10:
                res += self.generate_hint_boats(row, col + 1, "R", max_length)
            res = [e for e in res if len(e) > 2]
            for boat in res:
                can_put = True
                for piece in boat:
                    if 0 <= piece[0] < 10 and 0 <= piece[1] < 10:
                        pass
                    else:
                        can_put = False
                        break
                if can_put:
                    boat_coords.append(tuple(boat))
            boat_coords = list(set(boat_coords))
        else:
            for i in range(max_length - 1):
                coords = []
                if val == "T" and row + i + 1 < 10:
                    coords = [(row + k, col, tb[i][k]) if tb[i][k].upper() != self.get_value(row + k, col) else (
                        row + k, col, tb[i][k].upper()) for k in range(i + 2)]

                elif val == "B" and row - i - 1 >= 0:
                    coords = [
                        (row - i - 1 + k, col, tb[i][k]) if tb[i][k].upper() != self.get_value(row - i - 1 + k,
                                                                                               col) else (
                            row - i - 1 + k, col, tb[i][k].upper()) for k in range(i + 2)]
                elif val == "L" and col + i + 1 < 10:
                    coords = [(row, col + k, lr[i][k]) if lr[i][k].upper() != self.get_value(row, col + k) else (
                        row, col + k, lr[i][k].upper()) for k in range(i + 2)]
                elif val == "R" and col - i - 1 >= 0:
                    coords = [
                        (row, col - i - 1 + k, lr[i][k]) if lr[i][k].upper() != self.get_value(row,
                                                                                               col - i - 1 + k) else (
                            row, col - i - 1 + k, lr[i][k].upper()) for k in range(i + 2)]

                if coords != [] and (self.get_value(coords[-1][0], coords[-1][1]) == "M" or self.get_value(coords[0][0],
                                                                                                           coords[0][
                                                                                                               1]) == "M"):
                    coords = []
                if coords:
                    can_put = True
                    for cor in coords:
                        if not cor[2].isupper():
                            if (self.rows[cor[0]] - self.boats_placed_row[cor[0]] >= 1) and (
                                    self.columns[cor[1]] - self.boats_placed_col[cor[1]] >= 1):
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
                                        elif (
                                                key == "b"
                                                and value[1] == "B"
                                                and cor[2] == "t"
                                        ):
                                            pass
                                        elif (
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
                                        elif (
                                                key == "l"
                                                and value[1] == "L"
                                                and cor[2] == "r"
                                        ):
                                            pass
                                        elif (
                                                key == "r"
                                                and value[1] == "R"
                                                and cor[2] == "l"
                                        ):
                                            pass
                                        elif (
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
                                can_put = False
                                break
                    if can_put:
                        boat_coords.append(coords)
        return boat_coords

    def decrease_hint_boats(self):
        """Deteta os barcos que já foram colocados e
        subtrai-os aos barcos disponíveis"""
        self.hints = [hint for hint in self.hints if hint[2] not in ["W", "C"]]
        tested_coords = []
        for (row, col), value in np.ndenumerate(self.board):
            if (row, col) not in tested_coords:
                tested_coords.append((row, col))
                k = 0
                stop = False
                if value == "C":
                    self.available_boats = self.decrease_available_boats(1)
                elif value == "T":
                    while k < 3 and not stop:
                        v_vals = self.adjacent_vertical_values(row + k, col)
                        if v_vals[1] == "" or v_vals[1] == "W":
                            stop = True
                        elif v_vals[1] == "B":
                            coords = [(row + j, col) for j in range(0, k + 2)]
                            for coor in coords:
                                self.hints.remove((coor[0], coor[1], self.get_value(coor[0], coor[1])))
                            tested_coords.extend(
                                coords
                            )
                            self.available_boats = self.decrease_available_boats(k + 2)
                            stop = True
                        k += 1
                elif value == "L":
                    while k < 3 and not stop:
                        h_vals = self.adjacent_horizontal_values(row, col + k)
                        if h_vals[1] == "" or h_vals[1] == "W":
                            stop = True
                        elif h_vals[1] == "R":
                            coords = [(row, col + j) for j in range(1, k + 2)]
                            for coor in coords:
                                self.hints.remove((coor[0], coor[1], self.get_value(coor[0], coor[1])))
                            tested_coords.extend(
                                [(row, col + j) for j in range(1, k + 2)]
                            )
                            self.available_boats = self.decrease_available_boats(k + 2)
                            stop = True
                        k += 1

    def set_value(self, row: int, col: int, value: str):
        """Muda a board nas coordenadas dadas para value"""
        self.board[(row, col)] = value

    def get_adjacent_values(
            self, row: int, col: int
    ):
        """Retorna um dicionário com todas as coordenadas adjacentes e os seus respetivos valores"""
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
        for k, v in adjacent_coords.items():
            if 0 <= v[0] < 10 and 0 <= v[1] < 10:
                res[k] = (v, self.board[v])
        return res

    def adjacent_vertical_values(self, row: int, col: int):
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

    def adjacent_horizontal_values(self, row: int, col: int):
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

    # Posições ocupadas ao meter um barco
    def n_spaces_to_fill(self, action):
        """Conta o número de espaços que vão ser preenchidos por águas e barcos numa certa ação"""
        n = 0
        n += action["boat_length"]
        tested_pos = []

        if action["coords"][0][0] == action["coords"][1][0]:
            if self.rows[action["coords"][0][0]] - self.boats_placed_row[action["coords"][0][0]] - action[
                "boat_length"] == 0:
                for col, val in np.ndenumerate(self.board[action["coords"][0][0]]):
                    tested_pos.append((action["coords"][0][0], col[0]))
                    if val == "":
                        n += 1
        elif action["coords"][0][1] == action["coords"][1][1]:
            if self.rows[action["coords"][0][1]] - self.boats_placed_row[action["coords"][0][1]] - action[
                "boat_length"] == 0:
                for row, val in np.ndenumerate(self.board[:, action["coords"][0][1]]):
                    tested_pos.append((row, action["coords"][0][1]))
                    if val == "":
                        n += 1

        for coord in action["coords"]:
            adj_values = self.get_adjacent_values(coord[0], coord[1])
            for val in adj_values.values():
                if val[0] not in tested_pos and val[0] not in action["coords"]:
                    tested_pos.append(val[0])
                    if val[1] == "":
                        n += 1
        return n

    def print(self):
        """Dá print na board final"""
        res = ""
        for row in self.board:
            new_row = [""] * 10
            for i in range(10):
                if row[i] == "":
                    new_row[i] = "v"
                else:
                    new_row[i] = row[i]
            if res == "":
                res += "".join(new_row)
            else:
                res += "\n" + "".join(new_row)
        print(res)

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
        res["hints"] = []
        board = np.full([10, 10], "", dtype=str)
        for line in sys.stdin:
            split_line = line.split("\t")
            if split_line[0] == "ROW":
                # Mudar para np arrays?
                res["rows"] = [int(e) for e in split_line[1:]]
            elif split_line[0] == "COLUMN":
                res["columns"] = [int(e) for e in split_line[1:]]
            elif split_line[0] == "HINT":
                board[(int(split_line[1]), int(split_line[2]))] = split_line[-1]
                res["hints"].append((int(split_line[1]), int(split_line[2]), split_line[3][0]))
            else:
                res["n_hints"] = int(split_line[0])
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


class Bimaru(Problem):
    def __init__(self, boardS: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(boardS)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []

        if state.boardState.hints:
            for hint in state.boardState.hints:
                boat_is = "m"
                if hint in ["T", "B"]:
                    boat_is = "v"
                elif hint in ["L", "R"]:
                    boat_is = "h"
                hint_boats = state.boardState.generate_hint_boats(hint[0], hint[1], hint[2])
                for boat in hint_boats:
                    if boat_is == "v" and state.boardState.columns[boat[0][1]] - state.boardState.boats_placed_col[boat[0][1]] >= len(boat) - 1:
                        actions.append({"coords": boat, "boat_length": len(boat), "hint": hint})
                    elif boat_is == "h" and state.boardState.rows[boat[0][0]] - state.boardState.boats_placed_row[boat[0][0]] >= len(boat) - 1:
                        actions.append({"coords": boat, "boat_length": len(boat), "hint": hint})
                    elif boat_is == "m":
                        if boat[0][0] == boat[1][0] and state.boardState.rows[boat[0][0]] - \
                                state.boardState.boats_placed_row[boat[0][0]] >= len(boat) - 1:
                            actions.append({"coords": boat, "boat_length": len(boat), "hint": hint})
                        elif boat[0][1] == boat[1][1] and state.boardState.columns[boat[0][1]] - \
                                state.boardState.boats_placed_col[boat[0][1]] >= len(boat) - 1:
                            actions.append({"coords": boat, "boat_length": len(boat), "hint": hint})
        else:
            boat_length = 0
            for key in range(1, 5):
                if state.boardState.available_boats[key - 1] != 0:
                    boat_length = key

            for i in range(10):
                if state.boardState.rows[i] - state.boardState.boats_placed_row[i] >= boat_length:
                    boats = state.boardState.horizontal_boats(i, boat_length)
                    for boat in boats:
                        actions.append({"coords": boat, "boat_length": boat_length, "hint": None})

            for j in range(10):
                if state.boardState.columns[j] - state.boardState.boats_placed_col[j] >= boat_length:
                    boats = state.boardState.vertical_boats(j, boat_length)
                    for boat in boats:
                        actions.append({"coords": boat, "boat_length": boat_length, "hint": None})

        if state.boardState.hints == []:
            if boat_length > 2:
                actions.sort(reverse=True, key=state.boardState.n_spaces_to_fill)
        return actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        if action["hint"]:
            new_hints = [h for h in state.boardState.hints if h != action["hint"]]
        else:
            new_hints = None
        new_board = Board(board=state.boardState.set_boat(action["coords"]),
                          rows=state.boardState.rows,
                          columns=state.boardState.columns,
                          hints=new_hints,
                          available_boats=state.boardState.decrease_available_boats(action["boat_length"]))
        new_state = BimaruState(new_board)

        new_state.boardState.find_boats_and_empty_spots()

        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas conforme as regras do problema."""

        curr_board = state.boardState.board
        rows = state.boardState.rows
        cols = state.boardState.columns

        # if state.boardState.available_boats[1]==1:
        # print(state.boardState.rows, state.boardState.columns)
        # print(state.boardState.available_boats)
        # state.boardState.print()
        # print("----------------------------")

        for r_i in range(len(rows)):
            n_boats = sum(
                [1 for e in curr_board[r_i, :] if e != "" and e != "." and e != "W"]
            )
            if n_boats != rows[r_i]:
                return False

        for c_i in range(len(cols)):
            n_boats = sum(
                [1 for e in curr_board[:, c_i] if e != "" and e != "." and e != "W"]
            )
            if n_boats != cols[c_i]:
                return False

        if state.boardState.available_boats != [0, 0, 0, 0]:
            return False

        for (r, c), value in np.ndenumerate(curr_board):
            if curr_board[r, c] == "":
                return False
        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass


if __name__ == "__main__":
    parsed = Board.parse_instance()
    initial_board = Board(parsed["board"], parsed["rows"], parsed["columns"], parsed["hints"])
    initial_board.decrease_hint_boats()
    initial_board.set_initial_water()
    initial_board.find_boats_and_empty_spots()
    problem = Bimaru(initial_board)
    result = depth_first_tree_search(problem)
    result.state.boardState.print()
