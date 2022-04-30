from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    For the first 10 moves, we will play randomly, with the intention of breaking any competitor algorithm that uses 
    the first few moves to establish a set policy. After the first 10 moves, we will switch between randomly playing
    (30% of the time) and playing using a Bayesian policy. The Bayesian policy itself can be broken down into two 
    componenets. We keep track of number of times opponent plays rock, paper, or scissors given he plays each of 
    rock, paper, scissors, and a similar matrix for our own moves. In each of our turn, we compute 4 moves: one by 
    playing the best response to the opponent assuming he will play the move corresponding to the maximum number of
    times played before (i.e. if his last move was rock and according to our matrix he plays scissors next most 
    often, then counter by playing rock), another by playing the best response to a move randomly generated with 
    probabilities defined by frequency counts of opponents next moves given his last move. Similarly, we assume the
    opponenet is also keeping track of OUR matrix of future move given previous move, so we guess what he will play
    using the two methods just stated, and respond against it. For example, given our last move was rock, and our 
    next move according to previous moves is most likely to be rock again, the predict opponent detect that and 
    plays paper, so we play scissors. In the update_results function, we record scores for each of these strategies 
    by adding 1 if they would have won that round, else -1. Then, when making our move, we simply select the 
    strategy that has the highest score so far.
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        self.move_num = 0
        self.n_moves = game_matrix.shape[0]
        self.my_moves = []
        self.opp_moves = []
        self.best_move = {0:1, 1:2, 2:0}
        self.opp_is_firstplayer = False
        self.opp_is_copycat = False
        self.opp_is_goldfish = False

        self.bayesian_matrix = [[1,1,1], [1,1,1], [1,1,1]]
        self.my_bayesian_matrix = [[1,1,1], [1,1,1], [1,1,1]]

        self.scores = [0, 0, 0, 0] # bayesian score, my_bayesian score, and their deterministic versions
        self.moves = [0,0,0,0]

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        def opp_goldfish():
            for i in range(1, len(self.opp_moves)):
                if self.opp_moves[i] != self.best_move[self.my_moves[i-1]]:
                    return False
            return True

        if self.move_num < 10:
            return np.random.randint(0, self.n_moves)

        elif self.move_num == 10:
            self.opp_is_goldfish = opp_goldfish()
            return np.random.randint(0, self.n_moves)

        else:
            # first player
            if len(set(self.opp_moves[-10:])) == 1:
                return self.best_move[self.opp_moves[-1]]
            
            # copycat
            elif  self.my_moves[-10:-1] == self.opp_moves[-9:]:
                return self.best_move[self.my_moves[-1]]
            
            # goldfish
            elif self.opp_is_goldfish:
                return self.best_move[self.best_move[self.my_moves[-1]]]

            else:
                if np.random.uniform(0, 1) < 0.3: # with 30% chance, play randomly
                    return np.random.randint(0, self.n_moves)
                else: # bayesian strategy
                    strat = np.argmax(self.scores)
                    rock = self.bayesian_matrix[self.opp_moves[-1]][0]
                    paper = self.bayesian_matrix[self.opp_moves[-1]][1]
                    scissors = self.bayesian_matrix[self.opp_moves[-1]][2]
                    count = rock + paper + scissors

                    move2 = self.best_move[np.argmax([rock, paper, scissors])]
                    move0 = self.best_move[np.random.choice(3, p=[rock/count, paper/count, scissors/count])]

                    my_rock = self.my_bayesian_matrix[self.my_moves[-1]][0]
                    my_paper = self.my_bayesian_matrix[self.my_moves[-1]][1]
                    my_scissors = self.my_bayesian_matrix[self.my_moves[-1]][2]
                    my_count = my_rock + my_paper + my_scissors

                    move3 = self.best_move[self.best_move[np.argmax([my_rock, my_paper, my_scissors])]]
                    move1 = self.best_move[self.best_move[np.random.choice(3, p=[my_rock/my_count, my_paper/my_count, my_scissors/my_count])]]

                    if strat == 0:
                        return move0
                    elif strat == 1:
                        return move1
                    elif strat == 2:
                        return move2
                    else:
                        return move3


    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # bayesian best strategy
        if self.move_num >= 10:
            rock = self.bayesian_matrix[self.opp_moves[-1]][0]
            paper = self.bayesian_matrix[self.opp_moves[-1]][1]
            scissors = self.bayesian_matrix[self.opp_moves[-1]][2]
            count = rock + paper + scissors
            move2 = self.best_move[np.argmax([rock, paper, scissors])]
            move0 = self.best_move[np.random.choice(3, p=[rock/count, paper/count, scissors/count])]

            my_rock = self.my_bayesian_matrix[self.my_moves[-1]][0]
            my_paper = self.my_bayesian_matrix[self.my_moves[-1]][1]
            my_scissors = self.my_bayesian_matrix[self.my_moves[-1]][2]
            my_count = my_rock + my_paper + my_scissors
            move3 = self.best_move[self.best_move[np.argmax([my_rock, my_paper, my_scissors])]]
            move1 = self.best_move[self.best_move[np.random.choice(3, p=[my_rock/my_count, my_paper/my_count, my_scissors/my_count])]]

            if move0 == self.best_move[other_move]:
                self.scores[0] += 1
            else:
                self.scores[0] -= 1
            if move1 == self.best_move[other_move]:
                self.scores[1] += 1
            else:
                self.scores[1] -= 1
            if move2 == self.best_move[other_move]:
                self.scores[2] += 1
            else:
                self.scores[2] -= 1
            if move3 == self.best_move[other_move]:
                self.scores[3] += 1
            else:
                self.scores[3] -= 1

        if self.my_moves:
            self.bayesian_matrix[self.opp_moves[-1]][other_move] += 1
            self.my_bayesian_matrix[self.my_moves[-1]][my_move] += 1

        self.my_moves.append(my_move)
        self.opp_moves.append(other_move)
        self.move_num += 1


    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        self.move_num = 0
        self.my_moves = []
        self.opp_moves = []
        self.best_move = {0:1, 1:2, 2:0}
        self.opp_is_firstplayer = False
        self.opp_is_copycat = False
        self.opp_is_goldfish = False

        self.bayesian_matrix = [[1,1,1], [1,1,1], [1,1,1]]
        self.my_bayesian_matrix = [[1,1,1], [1,1,1], [1,1,1]]

        self.scores = [0, 0, 0, 0] # bayesian score, my_bayesian score, and their deterministic versions
        self.moves = [0,0,0,0]

if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))
