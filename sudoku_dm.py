"""

Create by: jesseclark
On: 10/10/16

Solve sudoku using projection onto sets methods.
http://www.pnas.org/content/104/2/418.full
'Searching with iterated maps'
V. Elser, I. Rankenburg, and P. Thibault
vol. 104 no. 2, 2007

"""

import numpy as np
from collections import Counter
from PIL import Image,ImageFont,ImageDraw
import scipy.misc


def check_solution(board):
    """
    Check if the board is a solution
    :param board: nxn board as a numpy array
    :return: bool
    """

    # check rows are complete set
    for ii in range(board.shape[0]):
        if not sorted(list(board[ii,:])) == range(1,board.shape[0]+1):
            return False

    # check columns are a complete set
    for jj in range(board.shape[1]):
        if not sorted(list(board[:,jj])) == range(1,board.shape[1]+1):
            return False

    # now check that each 3x3 sub square is a complete set
    for yy in range(3):
        for xx in range(3):
            # get the sub block
            temp = board[(yy*3):(yy+1)*3,(xx*3):(xx+1)*3]
            if Counter(temp.reshape(temp.size)) != Counter(range(1,10)):
                return False

    return True


def PC1(Q):
    """
    First projection
    :param Q: nxnxn numpy array
    :return: nxnxn numpy array
    """

    n = Q.shape[0]
    P = np.zeros(Q.size)
    seq = np.array(range(0,n))

    for i in range(0,n):
        for k in range(0,n):
            ix = k*n*n + n*seq + i
            temp = Q.copy().reshape(n*n*n)
            order = np.argsort(temp[ix])[::-1]
            #P[i, order[0], k] = 1
            #P[order[0],i, k] = 1
            P[ix[order[0]]] = 1

    return P.reshape(Q.shape)


def RC1(Q):
    """
    First reflecection
    :param Q: nxnxn numpy array
    :return: nxnxn numpy array
    """
    return 2*PC1(Q) - Q


def PC2(Q):
    """
    Second projection
    :param Q: nxnxn numpy array
    :return: nxnxn numpy array
    """

    n = Q.shape[0]
    P = np.zeros(Q.size)
    seq = np.array(range(0,n))

    for j in range(0,n):
        for k in range(0,n):
            ix = k*n*n + j*n + seq
            temp = Q.reshape(1,n*n*n)[0]
            order = np.argsort(temp[ix])[::-1]
            #P[order[0],j, k] = 1
            P[ix[order[0]]] = 1

    return P.reshape(Q.shape)


def RC2(Q):
    """
    Second reflection
    :param Q: nxnxn numpy array
    :return: nxnxn numpy array
    """
    return 2*PC2(Q) - Q


def PC3(Q):
    """
    Third projection
    :param Q: nxnxn numpy array
    :return: nxnxn numpy array
    """

    n = Q.shape[0]
    P = np.zeros((n*n,n))
    blocksize = int(np.sqrt(n))
    mask = np.zeros((blocksize*blocksize))

    for i in range(0,blocksize):
        for j in range(0,blocksize):
            mask[i+blocksize*j] = i + n*j

    #P = P.reshape(n*n*n)
    for k in range(0,n):
        for i in range(0, blocksize):
            for j in range(0, blocksize):
                ix = (mask + (i*blocksize + j*n*blocksize)).astype(int)
                temp = Q.reshape(n*n,n)
                #print(temp[ix,k].sum())
                order = np.argsort(temp[ix,k])[::-1]
                #P[ix[order[0]],j, k] = 1
                P[ix[order[0]],k] = 1

    return P.reshape(Q.shape)


def RC3(Q):
    return 2*PC3(Q) - Q


def PC4(Q):

    n = Q.shape[0]
    P = np.zeros(Q.size)
    seq = np.array(range(0,n))

    for i in range(0,n):
        for j in range(0,n):
            ix = seq*n*n + j*n +i
            temp = Q.reshape(n*n*n)
            order = np.argsort(temp[ix])[::-1]
            P[ix[order[0]]] = 1
    return P.reshape(Q.shape)


def RC4(Q):
    return 2*PC4(Q) - Q


def PC5(Q, board):
    """
    Fifth projection - enforce pre-existing constraints
    :param Q: nxnxn numpy array
    :param board nxn array with 1-9 values
    :return: nxnxn numpy array
    """

    n = Q.shape[0]

    for i in range(0,n):
        for j in range(0,n):
            if board[i,j] != 0:
                Q[i,j,board[i,j]-1] = 1
    return Q



def RC5(Q, board):
    return 2*PC5(Q, board) - Q



def Q_to_board(Q, reverse=False):
    """
    Take the nxnxn sudoku array (0's and 1's) and turn it into an
    nxn array with the numbers 1-9
    :param Q: nxnxn sudoku array
    :param reverse: reverse the direction
    :return: the nxn array with vals of 1-9
    """

    n = Q.shape[0]
    board = np.zeros(Q.shape[:2])

    for i in range(0,n):
        for j in range(0,n):
            if not reverse:
                board[i,j] = np.argmax(Q[i,j,:])+1
            else:
                board[j,i] = np.argmax(Q[j,i,:])+1

    return board


def board_to_Q(board, reverse=False):
    """
    Take the nxn array with the numbers 1-9 and make the
    nxnxn sudoku array (0's and 1's)
    :param board: nxn sudoku board with vals of 1-9
    :param reverse: reverse the direction
    :return: the nxnxn array with each value vectorized
    """

    n = board.shape[0]
    Q = np.zeros((n,n,n))

    for i in range(0,n):
        for j in range(0,n):
            if not reverse:
                Q[i,j,int(board[i,j]-1)] = 1
            else:
                Q[j,i,int(board[j,i]-1)] = 1
    return Q


def P_avg(X_i):
    """ average projection
    """
    X_avg = np.mean(X_i,0)

    return np.stack([X_avg for ind in range(1,6)],0)


def P_i(X_i, board):

    # get the projections
    P_is = {0:PC1, 1:PC2, 2:PC3, 3:PC4, 4:PC5}

    return np.array([P_is[ind](X_i[ind]) if ind != 4 else P_is[ind](X_i[ind], board) for ind in range(0, X_i.shape[0])])


def boards_to_images(boards, save_dir='',
                   save_prefix='it-', n_out=256, font_size=20, extra=10):

    # add some to the end when the solution has been found
    if extra > 0:
        boards += [boards[-1] for ind in range(extra)]

    for ind,board in enumerate(boards):
        board_to_image(board, save_dir, save_prefix+str(ind).zfill(3), n_out, font_size, title=str(ind).zfill(3))


def board_to_image(board, save_dir='',
                   save_prefix='it-', n_out=256, font_size=20, title=None):

    # create a blank image for writing to
    blank = np.zeros([n_out,n_out,3])
    bname = 'blank.png'
    scipy.misc.imsave(save_dir+bname,blank)

    # the positions of the numbers
    offset = 5
    ints = np.array(range(0,n_out,n_out/10)[1:10])-offset

    # read in the blank board and write
    image = Image.open(save_dir+bname)
    draw  = ImageDraw.Draw(image)
    font  = ImageFont.truetype("/Library/Fonts/arial.ttf", font_size, encoding="unic")

    if title is not None:
        draw.text( (5,5), title, fill='#ffffff',
                   font=ImageFont.truetype("/Library/Fonts/arial.ttf", 10, encoding="unic"))


    # loop through each number and draw
    for xx in range(9):
        for yy in range(9):
            numb = board[yy,xx]
            draw.text( (ints[xx],ints[yy]), str(int(numb)), fill='#ffffff', font=font)

    # save the image
    image.save(save_dir+save_prefix+'.png',"PNG")


def get_preset_puzzle(numb=1):
    """
    Get a preset puzzle to solve
    :param numb: which puzzle to return
    :return: nxn board with 0's where the values are missing
    """

    puzzles = {1:np.array([
                [0, 0, 0, 7, 0, 0, 0, 8, 0],
                [0 ,9 ,0, 0, 0, 3, 1 ,0, 0 ],
                [0, 0, 6, 8, 0, 5, 0, 7, 0 ],
                [0, 2, 0, 6, 0, 0, 0, 4, 9 ],
                [0 ,0, 0, 2, 0, 0, 0, 5, 0 ],
                [0 ,0 ,8, 0, 4, 0, 0, 0, 7 ],
                [0 ,0, 0, 9, 0, 0, 0, 3, 0 ],
                [3 ,7, 0, 0, 0, 0, 0, 0, 6 ],
                [1 ,0, 5, 0, 0, 4 ,0 ,0, 0 ]]
                ),
               2:np.array( [
                [0 ,9 ,0, 0, 8, 0, 0, 4, 0],
                [7 ,0, 0, 3, 0, 9, 0, 0, 8 ],
                [0, 0 ,5 ,0, 0, 0, 3, 0, 0 ],
                [0 ,7, 0, 0, 0, 0, 0, 5, 0 ],
                [8 ,0, 0, 0, 2, 0, 0, 0, 6 ],
                [0 ,1, 0, 0, 0, 0, 0, 2 ,0 ],
                [0, 0, 9, 0, 0, 0, 7, 0, 0 ],
                [6 ,0 ,0, 2, 0, 1, 0, 0 ,5 ],
                [0 ,5 ,0 ,0, 3, 0, 0, 8, 0 ]]
                ),
               3:np.array([
                [4 ,8, 3, 9, 0, 1 ,6, 5, 7],
                [9, 6, 7, 3, 4, 5, 8, 2, 1],
                [2, 5, 1, 8, 7, 0, 4, 9, 3],
                [5 ,4, 8, 0, 3, 2, 9, 7, 0],
                [7 ,2, 9, 5, 6, 4, 1, 3, 8],
                [1 ,3, 6, 7, 0, 8, 2, 4, 5],
                [3, 7, 2, 0, 8, 9, 0, 1, 4],
                [8, 1 ,4, 2, 5, 3, 7, 6 ,9],
                [6, 9, 5 ,4 ,1, 7, 3, 8, 0]]
                ),
               4:np.array([
                [0 ,8, 3, 9, 2, 1 ,6, 5, 7],
                [9, 6, 7, 0, 4, 5, 8, 2, 1],
                [2, 5, 1, 8, 7, 6, 4, 9, 3],
                [5 ,4, 8, 0, 3, 0, 9, 7, 6],
                [0 ,2, 9, 5, 6, 4, 1, 3, 8],
                [1 ,0, 6, 7, 9, 8, 2, 4, 5],
                [3, 7, 2, 0, 8, 9, 5, 1, 4],
                [8, 0 ,4, 2, 5, 3, 7, 6 ,9],
                [6, 9, 5 ,4 ,1, 7, 3, 8, 0]]
                ),
               5:np.array([
                [0, 0, 4, 0, 7, 0, 0, 0, 8 ],
                [0, 0, 5, 0, 0, 6, 0, 0, 0],
                [6, 0, 0, 0, 0, 8, 0, 0, 3],
                [0 ,0 ,0 ,0, 9, 0, 0, 1, 7],
                [0, 0, 0, 0, 2, 0, 0, 0, 5],
                [9, 3, 0, 0, 0, 0, 6, 0, 0],
                [2, 0, 0, 0 ,5 ,0, 0, 0, 1],
                [0, 8, 0, 4, 0, 0, 0, 9, 0],
                [0, 7, 0, 0, 1, 0, 0, 8 ,0]]
                ),
               6:np.array([
                [8, 2, 0, 9, 5, 7, 0, 0, 0],
                [0, 1, 0, 0, 0, 2, 0, 0, 8],
                [0, 0, 0, 0, 0, 4, 2, 0, 0],
                [0 ,7 ,0 ,0, 1, 0, 4, 0, 0],
                [0, 8, 5, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 9, 0, 0, 6, 0],
                [0, 0, 7, 4 ,0 ,0, 0, 0, 0],
                [9, 0, 0, 6, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 9, 0, 0 ,3]]
                )}

    return puzzles[numb]



if __name__ == "__main__":

    # select which pre-set puzzle to use
    puzzle_numb = 6

    # get the preset puzzle
    board = get_preset_puzzle(puzzle_numb)

    # max number of iterations
    max_iterations = 1000

    # get the board size
    n = board.shape[0]

    # make the board tensor
    Q = board_to_Q(board)

    # init the board
    X_0 = np.random.randint(0,2,(n,n,n))

    # get the X_i's
    X_i = np.stack([X_0 for ind in range(1,6)],0)

    # Average projection operator
    PB_X = P_avg(X_i)

    # store errors and iterates
    errors = []
    iterates = []

    for ii in range(0, max_iterations):

        # check the solution
        if check_solution(Q_to_board(PB_X[0])):
            print('Solution',ii,error)
            break

        # difference map update
        # x' = x + P_i(2PB(x) - x) - PB(x)

        # get the average projection
        PB_X = P_avg(X_i)
        # get the difference of projected reflection and average projection
        D_X = P_i(2*PB_X-X_i, board) - PB_X

        # update
        X_i = X_i + D_X

        # error between succesive projections
        error = np.abs(PB_X-P_avg(X_i)).mean()

        errors.append(error)
        iterates.append(Q_to_board(PB_X[0]))
        print(error)

    # convert from nxnxn to nxn board
    soln = Q_to_board(PB_X[0])
    print('done [{ii}]'.format(ii=ii))
    print(check_solution(soln))
    print(soln)
    print(soln.sum(0))
    print(soln.sum(1))


    # output images of the solution to make a movie of the process
    boards_to_images(iterates, save_prefix=str(puzzle_numb)+'-')


