# This is Player class that represent player in Gomoku game
# implemented minimax algorithm, alpha beta pruning alghoritm
# author: Piotr Obara

import numpy as np
import misc
import math

from misc import legalMove
from misc import winningTest
from misc import rowTest
from misc import diagTest
from gomokuAgent import GomokuAgent

class Player(GomokuAgent):


    def move(self, board):

        maxPlayer = self.ID

        moveLoc = self.minimax(board, 4, -math.inf, +math.inf, True)[0]

        return moveLoc

    # minimax function with Alpha-Beta Pruning
    def minimax(self, currentBoard, depth, alpha, beta, isMaxPlayer):
        maxPlayerID = self.ID
        if self.ID == 1:
            minPlayerID = -1
        else:
            minPlayerID = 1

        # check if depth is 0 or either player won or there is no more valid moves
        if depth == 0 or winningTest(maxPlayerID, currentBoard, self.X_IN_A_LINE) or winningTest(minPlayerID, currentBoard, self.X_IN_A_LINE):
            if winningTest(maxPlayerID, currentBoard, self.X_IN_A_LINE) or winningTest(minPlayerID, currentBoard, self.X_IN_A_LINE):
                if winningTest(maxPlayerID, currentBoard, self.X_IN_A_LINE):
                    return (None, 100000000000000)
                if winningTest(minPlayerID, currentBoard, self.X_IN_A_LINE):
                    return (None, -100000000000000)
                else: #No more valid moves
                    return (None, 0)
            else:
                return (None, evaluationFunction(currentBoard, self.X_IN_A_LINE, maxPlayerID, minPlayerID))

        # if isMaxPlayer = True
        if isMaxPlayer:
            maxEvaluation = -math.inf
            childStates = createChildsStates(currentBoard, maxPlayerID, minPlayerID, self.X_IN_A_LINE)

            bestPosition = getPosition(currentBoard, self.BOARD_SIZE)

            for childState in childStates:
                tempBoard = np.array(currentBoard)
                tempBoard[childState[0],childState[1]] = maxPlayerID
                eval = self.minimax(tempBoard, depth -  1, alpha, beta, False)[1]
                # maxEvaluation = max(maxEvaluation, eval)
                if eval > maxEvaluation:
                    maxEvaluation = eval
                    bestPosition = childState
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return bestPosition, maxEvaluation

        # if isMaxPlayer = False
        else:
            minEvaluation = math.inf
            childStates = createChildsStates(currentBoard, minPlayerID, minPlayerID, self.X_IN_A_LINE)
            bestPosition = getPosition(currentBoard, self.BOARD_SIZE)
            for childState in childStates:
                tempBoard = np.array(currentBoard)
                tempBoard[childState[0], childState[1]] = minPlayerID
                eval = self.minimax(tempBoard, depth - 1, alpha, beta, True)[1]
                if eval < minEvaluation:
                    minEvaluation = eval
                    bestPosition = childState
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return bestPosition, minEvaluation



# function score given state on board based on longest players chains
def evaluationFunction(board, xInLine, maxPlayerID, minPlayerID):
    bestMax = checkMaxPlayer(board, xInLine, maxPlayerID)
    bestMin = checkMinPlayer(board, xInLine, minPlayerID)

    result = bestMax - bestMin

    return result

# function count Max Player chain
def checkMaxPlayer(board, xInLine, maxPlayerID):
    best = 0

    for x in range(1, xInLine + 1):
        if winningTest(maxPlayerID, board, x):
            best = x
    return best

# function count Min Player chain
def checkMinPlayer(board, xInLine, minPlayerID):
    best = 0

    for x in range(1, xInLine + 1):
        if winningTest(minPlayerID, board, x):
            best = x
    return best

# function create smarts moves for current State. Return array of moves
def createChildsStates(board, currentPlayer, oponent, xInLine):
    BOARD_SIZE = board.shape[0]
    tempBoard = np.array(board)

    childsBoards = []

    currentPlayerChain = checkMaxPlayer(board, xInLine, currentPlayer)
    oponentPlayerChain = checkMinPlayer(board, xInLine, oponent)

    # if currentPlayer play first
    if checkIfEmpty(tempBoard, BOARD_SIZE, currentPlayer, oponent):

        moveLoc = tuple(np.random.randint(BOARD_SIZE, size=2))


        y = int(BOARD_SIZE / 2)
        x = int(BOARD_SIZE / 2)
        if legalMove(tempBoard, (moveLoc)):
            #tempBoard[BOARD_SIZE / 2, BOARD_SIZE / 2] = currentPlayer DELETE
            y = BOARD_SIZE/2
            x = BOARD_SIZE/2
            childsBoards.append((moveLoc))

    # if currentPlayer play second
    elif checkIfOponentOne(tempBoard, BOARD_SIZE, currentPlayer, oponent):
        loc = checkWhere(tempBoard, BOARD_SIZE, currentPlayer, oponent)

        y = loc[0]
        x = loc[1]

        if legalMove(tempBoard, (y,x)):
            # tempBoard[y, x] = currentPlayer DELETE
            childsBoards.append((y,x))

    # current player choose offense when have longer chain, defense otherwise
    elif oponentPlayerChain <= currentPlayerChain:
        if currentPlayerChain == 1:
            loc = findPlayer(tempBoard, currentPlayer, BOARD_SIZE)
            yy = loc[0]
            xx = loc[1]

            for y in range(3):
                for x in range(3):
                    if legalMove(tempBoard, (yy+y, xx+x)):

                        childsBoards.append((yy+y,xx+x))


        elif currentPlayerChain == 2 and rowTest(currentPlayer, tempBoard, 2):
            loc = getEndRow(currentPlayer, tempBoard, 2)
            yy = loc[0]
            xx = loc[1]

            yy -= 1
            xx -= 2

            if yy < 0:
                yy += 1
            if xx < 0:
                xx = 0

            # create boards around
            for y in range(3):
                for x in range(4):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif currentPlayerChain == 3 and rowTest(currentPlayer, tempBoard, 3):
            loc = getEndRow(currentPlayer, tempBoard, 3)
            yy = loc[0]
            xx = loc[1]
            yy -= 1
            xx -= 3

            if yy < 0:
                yy += 1
            if xx < 0:
                xx = 0

            for y in range(3):
                for x in range(5):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif currentPlayerChain == 4 and rowTest(currentPlayer, tempBoard, 4):
            loc = getEndRow(currentPlayer, tempBoard, 4)
            yy = loc[0]
            xx = loc[1]
            xx += 1

            y = loc[0]
            x = loc[1]
            x -= 4

            if legalMove(tempBoard, (yy, xx)):
                childsBoards.append((yy,xx))
            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y,x))


        elif currentPlayerChain == 2 and diagTest(currentPlayer, tempBoard, 2):
            loc = getEndDiag(currentPlayer, tempBoard, 2)

            y = loc[0]
            x = loc[1]
            x += 1
            y += 1

            if legalMove(tempBoard, (y,x)):
                childsBoards.append((y,x))

            z = loc[0]
            z -= 3
            w = loc[1]
            w -= 3

            if legalMove(tempBoard, (z,w)):
                childsBoards.append((z, w))

        elif currentPlayerChain == 3 and diagTest(currentPlayer, tempBoard, 3):
            loc = getEndDiag(currentPlayer, tempBoard, 3)
            y = loc[0]
            x = loc[1]
            y += 1
            y += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z -= 4
            w -= 4

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

        elif currentPlayerChain == 4 and diagTest(currentPlayer, tempBoard, 4):
            loc = getEndDiag(currentPlayer, tempBoard, 4)
            y = loc[0]
            x = loc[1]
            y += 1
            x += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z -= 5
            w -= 5

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

        elif currentPlayerChain == 2 and columnTestCustom(currentPlayer, tempBoard, 2):
            loc = getEndColumn(currentPlayer, tempBoard, 2)
            yy = loc[0]
            xx = loc[1]

            yy -= 2
            xx -= 1

            if yy < 0:
                yy += 2
            if xx < 0:
                xx =+ 1

            #create boards around
            for y in range(4):
                for x in range(3):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif currentPlayerChain == 3 and columnTestCustom(currentPlayer, tempBoard, 3):
            loc = getEndColumn(currentPlayer, tempBoard, 3)
            yy = loc[0]
            xx = loc[1]
            yy -= 3
            xx -= 1

            if yy < 0:
                yy += 1
            if xx < 0:
                xx = 0

            for y in range(5):
                for x in range(3):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif currentPlayerChain == 4 and columnTestCustom(currentPlayer, tempBoard, 4):
            loc = getEndColumn(currentPlayer, tempBoard, 4)
            flag = False

            yy = loc[0]
            xx = loc[1]
            yy += 1

            y = loc[0]
            x = loc[1]
            y -= 4

            if legalMove(tempBoard, (yy, xx)):
                childsBoards.append((yy,xx))
                flag = True
            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y,x))
                flag = True

        elif currentPlayerChain == 2 and diagTestCustom2(currentPlayer, tempBoard, 2):
            loc = getEndDiag2(currentPlayer, tempBoard, 2)

            y = loc[0]
            x = loc[1]
            x -= 1
            y += 1

            if legalMove(tempBoard, (y,x)):
                childsBoards.append((y,x))

            z = loc[0]
            z += 3
            w = loc[1]
            w -= 3

            if legalMove(tempBoard, (z,w)):
                childsBoards.append((z, w))

        elif currentPlayerChain == 3 and diagTestCustom2(currentPlayer, tempBoard, 3):
            loc = getEndDiag2(currentPlayer, tempBoard, 3)
            y = loc[0]
            x = loc[1]
            y -= 1
            y += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z += 4
            w -= 4

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

        elif currentPlayerChain == 4 and diagTestCustom2(currentPlayer, tempBoard, 4):
            loc = getEndDiag2(currentPlayer, tempBoard, 4)
            y = loc[0]
            x = loc[1]
            y -= 1
            x += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z += 5
            w -= 5

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

    elif oponentPlayerChain > currentPlayerChain:
        if oponentPlayerChain == 1:
            loc = findPlayer(tempBoard, oponent, BOARD_SIZE)
            yy = loc[0]
            xx = loc[1]

            for y in range(3):
                for x in range(3):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif oponentPlayerChain == 2 and rowTest(oponent, tempBoard, 2):
            loc = getEndRow(oponent, tempBoard, 2)
            yy = loc[0]
            xx = loc[1]

            yy -= 1
            xx -= 2

            if yy < 0:
                yy += 1
            if xx < 0:
                xx = 0

            #create boards around
            for y in range(3):
                for x in range(4):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif oponentPlayerChain == 3 and rowTest(oponent, tempBoard, 3):
            loc = getEndRow(oponent, tempBoard, 3)
            yy = loc[0]
            xx = loc[1]
            yy -= 1
            xx -= 3

            if yy < 0:
                yy += 1
            if xx < 0:
                xx = 0

            for y in range(3):
                for x in range(5):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif oponentPlayerChain == 4 and rowTest(oponent, tempBoard, 4):
            loc = getEndRow(oponent, tempBoard, 4)
            yy = loc[0]
            xx = loc[1]
            xx += 1

            y = loc[0]
            x = loc[1]
            x -= 4

            if legalMove(tempBoard, (yy, xx)):
                childsBoards.append((yy,xx))
            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y,x))


        elif oponentPlayerChain == 2 and diagTest(oponent, tempBoard, 2):
            loc = getEndDiag(oponent, tempBoard, 2)

            y = loc[0]
            x = loc[1]
            x += 1
            y += 1

            if legalMove(tempBoard, (y,x)):
                childsBoards.append((y,x))

            z = loc[0]
            z -= 3
            w = loc[1]
            w -= 3

            if legalMove(tempBoard, (z,w)):
                childsBoards.append((z, w))

        elif oponentPlayerChain == 3 and diagTest(oponent, tempBoard, 3):
            loc = getEndDiag(oponent, tempBoard, 3)
            y = loc[0]
            x = loc[1]
            y += 1
            y += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z -= 4
            w -= 4

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

        elif oponentPlayerChain == 4 and diagTest(oponent, tempBoard, 4):
            loc = getEndDiag(oponent, tempBoard, 4)
            y = loc[0]
            x = loc[1]
            y += 1
            x += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z -= 5
            w -= 5

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

        elif oponentPlayerChain == 2 and columnTestCustom(oponent, tempBoard, 2):
            loc = getEndColumn(oponent, tempBoard, 2)
            yy = loc[0]
            xx = loc[1]

            yy -= 2
            xx -= 1

            if yy < 0:
                yy += 2
            if xx < 0:
                xx =+ 1

            # create boards around
            for y in range(4):
                for x in range(3):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif oponentPlayerChain == 3 and columnTestCustom(oponent, tempBoard, 3):
            loc = getEndColumn(oponent, tempBoard, 3)
            yy = loc[0]
            xx = loc[1]
            yy -= 3
            xx -= 1

            if yy < 0:
                yy += 1
            if xx < 0:
                xx = 0

            for y in range(5):
                for x in range(3):
                    if legalMove(tempBoard, (yy+y, xx+x)):
                        childsBoards.append((yy+y,xx+x))

        elif oponentPlayerChain == 4 and columnTestCustom(oponent, tempBoard, 4):
            loc = getEndColumn(oponent, tempBoard, 4)
            flag = False

            yy = loc[0]
            xx = loc[1]
            yy += 1

            y = loc[0]
            x = loc[1]
            y -= 4

            if legalMove(tempBoard, (yy, xx)):
                childsBoards.append((yy,xx))
                flag = True
            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y,x))
                flag = True

        elif oponentPlayerChain == 2 and diagTestCustom2(oponent, tempBoard, 2):
            loc = getEndDiag2(oponent, tempBoard, 2)

            y = loc[0]
            x = loc[1]
            x -= 1
            y += 1

            if legalMove(tempBoard, (y,x)):
                childsBoards.append((y,x))

            z = loc[0]
            z += 3
            w = loc[1]
            w -= 3

            if legalMove(tempBoard, (z,w)):
                childsBoards.append((z, w))

        elif oponentPlayerChain == 3 and diagTestCustom2(oponent, tempBoard, 3):
            loc = getEndDiag2(oponent, tempBoard, 3)
            y = loc[0]
            x = loc[1]
            y -= 1
            y += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z += 4
            w -= 4

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

        elif oponentPlayerChain == 4 and diagTestCustom2(oponent, tempBoard, 4):
            loc = getEndDiag2(oponent, tempBoard, 4)
            y = loc[0]
            x = loc[1]
            y -= 1
            x += 1

            if legalMove(tempBoard, (y, x)):
                childsBoards.append((y, x))

            z = loc[0]
            w = loc[1]
            z += 5
            w -= 5

            if legalMove(tempBoard, (z, w)):
                childsBoards.append((z, w))

    return childsBoards

# function return given Player position
def findPlayer(board, currentPlayer, BOARD_SIZE):
    y = 0
    x = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r,c] == currentPlayer:
                y = r
                x = c
    return (y,x)

# function return last position of chain in row
def getEndRow(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    mask = np.ones(X_IN_A_LINE, dtype=int)*playerID

    loc = (0,0)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE-X_IN_A_LINE+1):
            flag = True
            for i in range(X_IN_A_LINE):

                if board[r,c+i] != playerID:
                    flag = False
                    break
            if flag:
                loc = (r,c+i)
                return loc

    return loc

# function return last position of chain in column
def getEndColumn(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    mask = np.ones(X_IN_A_LINE, dtype=int)*playerID

    loc = (0,0)
    for r in range(BOARD_SIZE-X_IN_A_LINE+2):
        for c in range(BOARD_SIZE):
            flag = True
            for i in range(X_IN_A_LINE):

                if board[r+i,c] != playerID:
                    flag = False
                    break
            if flag:
                loc = (r+i,c)
                return loc

    return loc

# function return last position of chain in diag
def getEndDiag(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    loc = (0, 0)
    for r in range(BOARD_SIZE - X_IN_A_LINE + 1):
        for c in range(BOARD_SIZE - X_IN_A_LINE + 1):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r+i,c+i] != playerID:
                    flag = False
                    break
            if flag:
                loc = r+i,c+i
                return loc
    return loc

# function return last position of chain in diag (diffrent direction)
def getEndDiag2(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    loc = (0, 0)
    for r in range(BOARD_SIZE - 1, X_IN_A_LINE -2, -1):
        for c in range(BOARD_SIZE - X_IN_A_LINE + 1):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r-i,c+i] != playerID:
                    flag = False
                    break
            if flag:
                loc = (r-i,c+i)
                return loc
    return loc

# function return True if chain exist False otherwise
def diagTestCustom(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    for r in range(BOARD_SIZE - 1, X_IN_A_LINE -2, -1):
        for c in range(BOARD_SIZE - X_IN_A_LINE + 1):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r-i,c+i] != playerID:
                    flag = False
                    break
            if flag:
                return True
    return False

# function return True if chain exist False otherwise
def diagTestCustom2(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    loc = (0, 0)
    for r in range(BOARD_SIZE - 1, X_IN_A_LINE -2, -1):
        for c in range(BOARD_SIZE - X_IN_A_LINE + 1):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r-i,c+i] != playerID:
                    flag = False
                    break
            if flag:
                return True
    return False

# function check if given board is empty
def checkIfEmpty(board, BOARD_SIZE, currentPlayer, oponent):
    flag = True
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r,c] == currentPlayer or board[r, c] == oponent:
                flag = False
                break

    return flag

# function check if there is only no currentPlayer element
def checkIfOponentOne(board, BOARD_SIZE, currentPlayer, oponent):
    flag = True
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r,c] == currentPlayer:
                flag = False
                break

    return flag





def checkWhere(board, BOARD_SIZE, currentPlayer, oponent):

    y = 0
    x = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r,c] == oponent:
                if r < BOARD_SIZE/2 and c < BOARD_SIZE/2:
                    y = r + 1
                    x = c + 1
                elif r < BOARD_SIZE/2 and c > BOARD_SIZE/2:
                    y = r + 1
                    x = c - 1
                elif r > BOARD_SIZE/2 and c < BOARD_SIZE/2:
                    y = r - 1
                    x = c + 1
                elif r > BOARD_SIZE/2 and c > BOARD_SIZE/2:
                    y = r - 1
                    x = c - 1
                else:
                    y = r
                    x = c + 1

    return (y,x)

def columnTestCustom(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    mask = np.ones(X_IN_A_LINE, dtype=int)*playerID

    for r in range(BOARD_SIZE-X_IN_A_LINE+1):
        for c in range(BOARD_SIZE):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r+i,c] != playerID:
                    flag = False
                    break
            if flag:
                return True

    return False

def getPosition(board, boardSize):
    while True:
        moveLoc = tuple(np.random.randint(boardSize, size=2))
        if legalMove(board, moveLoc):
            return moveLoc


