"""
    Environment to simulate a mahjong competion rules (official Chineese version, without flowers) game for machine learning
    Written by Rowan Rosenberg 2025
"""

#TODO: Implement agent, implement full scoring
import concurrent.futures
from enum import Enum
import random
from itertools import combinations

class Tile():
    # Represents tiles, with a suit and a number
    # Suits are 1 'Bamboo', 2 'Character', 3 'Dot', 4 'Wind', 5 'Dragon'
    # Numbers are 1-9 for Bamboo, Character, Dot, 1-4 for Wind, 1-3 for Dragon
    def __init__(self, suit, number):
        self.suit = suit
        self.number = number

    def __str__(self):
        return  TileType(self.suit).name + "-" + str(self.number)

    def __eq__(self, other):
        return self.suit == other.suit and self.number == other.number
    
    def __lt__(self, other):
        return self.suit < other.suit or (self.suit == other.suit and self.number < other.number)
    
    def __hash__(self):
        return hash((self.suit, self.number))
    
class TileType(Enum):
    B = 1
    C = 2
    D = 3
    Wi = 4
    Dr = 5

def create_tiles():
    # Returns a list of shuffled tiles for start of game
    tiles = []
    # Add normal tiles
    for suit in range(1,4):
        for number in range(1,10):
            tiles += [Tile(suit, number)] * 4
    # Add wind tiles
    for number in range(1,5):
        tiles += [Tile(4, number)] * 4
    # Add dragon tiles
    for number in range(1,4):
        tiles += [Tile(5, number)] * 4
    # Shuffle
    random.shuffle(tiles)
    return tiles

def kong_options(hand):
    # Check if player can kong from hand, returns list of tiles that can be konged
    options = []
    for tile in hand:
        if hand.count(tile) == 4 and tile not in options:
            options.append(tile)
    return options

def kong_onto_meld_options(hand, melds):
    # Check if player can kong onto exposed pong, returns list of tiles that can be konged
    options = []
    for meld in melds:
        type, meld_tile = classify_meld(meld)
        if type == 0:
            for tile in hand:
                if meld_tile == tile and tile not in options:
                    options.append(tile)
    return options

def partition_tiles(hand):
    # Divides the hand into sets and eyes
    # Candidates are all tiles of which there are more than 1
    eyes_candidates = []

    if not hand:
        return []
    
    for tile in hand:
        if (tile not in eyes_candidates) and hand.count(tile) > 1:
            eyes_candidates.append(tile)

    persistant_hand = hand[:]
    # Try making eyes from every possible tile
    for eye in eyes_candidates:
        hand = persistant_hand[:]
        hand.remove(eye)
        hand.remove(eye)

        sets = [[eye,eye]]
        if not hand:
            return sets
        
        if len(hand) % 3 != 0:
            return []
        
        num_sets = len(hand)//3
        hand.sort()
        
        # Start recursive partitioning 
        match num_sets:

            case 1:
                new_sets = _one_set(hand)
            case 2:
                new_sets = _two_sets(hand)
            case 3:
                new_sets = _three_sets(hand)
            case 4:
                new_sets = _four_sets(hand)

        sets += new_sets
        # Check if everything was used as a set
        if len(new_sets) == num_sets:
            return sets

    # Empty return upon failiure 
    return []

def _one_set(hand):
        
    temp = hand[:]
    new_sets = []
    can_pong = can_chow = False
    # See if pong can be removed
    for tile in temp:
        if temp.count(tile) >= 3:
            can_pong = True
            break
        
    # See if chow can be removed
    for tile in temp:
        if tile.suit < 4 and Tile(tile.suit,tile.number+1) in temp and Tile(tile.suit,tile.number+2) in temp:
            can_chow = True
            break
    

    # Remove whichever is possible (can only be one)
    if can_pong:
        new_sets.append([temp[0],temp[0],temp[0]])
        temp.remove(temp[0])
        temp.remove(temp[0])
        temp.remove(temp[0])

    elif can_chow:
        tile = temp[0]
        new_sets.append([tile,Tile(tile.suit, tile.number+1),Tile(tile.suit, tile.number+2)])
        temp.remove(tile)
        temp.remove(Tile(tile.suit, tile.number+1))
        temp.remove(Tile(tile.suit, tile.number+2))

    return new_sets

def _two_sets(hand):

    temp = hand[:]
    new_sets = []
    can_pong = can_chow = False
    # See if pong can be removed
    for tile in temp:
        if temp.count(tile) >= 3:
            can_pong = True
            break
        
    # See if chow can be removed
    for tile in temp:
        if tile.suit < 4 and Tile(tile.suit,tile.number+1) in temp and Tile(tile.suit,tile.number+2) in temp:
            can_chow = True
            break

    # Remove pong first if both possible (ensures soundness)
    if can_pong:
        for tile in temp:
            if temp.count(tile) >= 3:
                new_sets.append([tile,tile,tile])
                temp.remove(tile)
                temp.remove(tile)
                temp.remove(tile)
                break
    # Remove chow if no pong
    elif can_chow:
        for tile in temp:
            if tile.suit < 4 and Tile(tile.suit, tile.number+1) in temp and Tile(tile.suit, tile.number+2) in temp:
                new_sets.append([tile,Tile(tile.suit, tile.number+1),Tile(tile.suit, tile.number+2)])
                temp.remove(tile)
                temp.remove(Tile(tile.suit, tile.number+1))
                temp.remove(Tile(tile.suit, tile.number+2))

    return new_sets + _one_set(temp)

def _three_sets(hand):

    temp = hand[:]
    new_sets = []
    can_pong = can_chow = False
    # See if pong can be removed
    for tile in temp:
        if temp.count(tile) >= 3:
            can_pong = True
            break
        
    # See if chow can be removed
    for tile in temp:
        if tile.suit < 4 and Tile(tile.suit,tile.number+1) in temp and Tile(tile.suit,tile.number+2) in temp:
            can_chow = True
            break
    
    # If only one can be removed, remove and recurse
    if can_pong and not can_chow:
        for tile in temp:
            if temp.count(tile) >= 3:
                new_sets.append([tile,tile,tile])
                temp.remove(tile)
                temp.remove(tile)
                temp.remove(tile)
                break
        return new_sets + _two_sets(temp)
    
    # Remove chow if no pong
    elif can_chow and not can_pong:
        for tile in temp:
            if tile.suit < 4 and Tile(tile.suit, tile.number+1) in temp and Tile(tile.suit, tile.number+2) in temp:
                new_sets.append([tile,Tile(tile.suit, tile.number+1),Tile(tile.suit, tile.number+2)])
                temp.remove(tile)
                temp.remove(Tile(tile.suit, tile.number+1))
                temp.remove(Tile(tile.suit, tile.number+2))
        return new_sets + _two_sets(temp)
    
    # If both possible, try both
    elif can_pong and can_chow:
        
        # Try ponging
        for tile in temp:
            if temp.count(tile) >= 3:
                new_sets.append([tile,tile,tile])
                temp.remove(tile)
                temp.remove(tile)
                temp.remove(tile)
                break
        new_sets +=  _two_sets(temp)

        # Check success
        if len(new_sets) == len(hand)/3:
            return new_sets
        
        # Reset and try chow
        temp = hand[:]
        new_sets = []
        
        for tile in temp:
            if tile.suit < 4 and Tile(tile.suit, tile.number+1) in temp and Tile(tile.suit, tile.number+2) in temp:
                new_sets.append([tile,Tile(tile.suit, tile.number+1),Tile(tile.suit, tile.number+2)])
                temp.remove(tile)
                temp.remove(Tile(tile.suit, tile.number+1))
                temp.remove(Tile(tile.suit, tile.number+2))
        new_sets += _two_sets(temp)

        # Check success
        if len(new_sets) == len(hand)/3:
            return new_sets

    return []

def _four_sets(hand):

    temp = hand[:]
    new_sets = []
    can_pong = can_chow = False
    # See if pong can be removed
    for tile in temp:
        if temp.count(tile) >= 3:
            can_pong = True
            break
        
    # See if chow can be removed
    for tile in temp:
        if tile.suit < 4 and Tile(tile.suit,tile.number+1) in temp and Tile(tile.suit,tile.number+2) in temp:
            can_chow = True
            break

    # If only one can be removed, remove and recurse
    if can_pong and not can_chow:
        for tile in temp:
            if temp.count(tile) >= 3:
                new_sets.append([tile,tile,tile])
                temp.remove(tile)
                temp.remove(tile)
                temp.remove(tile)
                break
        return new_sets + _three_sets(temp)
    
    # Remove chow if no pong
    elif can_chow and not can_pong:
        for tile in temp:
            if tile.suit < 4 and Tile(tile.suit, tile.number+1) in temp and Tile(tile.suit, tile.number+2) in temp:
                new_sets.append([tile,Tile(tile.suit, tile.number+1),Tile(tile.suit, tile.number+2)])
                temp.remove(tile)
                temp.remove(Tile(tile.suit, tile.number+1))
                temp.remove(Tile(tile.suit, tile.number+2))
        return new_sets + _three_sets(temp)
    
    # If both possible, try both
    elif can_pong and can_chow:
        
        # Try ponging
        for tile in temp:
            if temp.count(tile) >= 3:
                new_sets.append([tile,tile,tile])
                temp.remove(tile)
                temp.remove(tile)
                temp.remove(tile)
                break
        new_sets +=  _three_sets(temp)

        # Check success
        if len(new_sets) == len(hand)/3:
            return new_sets
        
        # Reset and try chow
        temp = hand[:]
        new_sets = []
        
        for tile in temp:
            if tile.suit < 4 and Tile(tile.suit, tile.number+1) in temp and Tile(tile.suit, tile.number+2) in temp:
                new_sets.append([tile,Tile(tile.suit, tile.number+1),Tile(tile.suit, tile.number+2)])
                temp.remove(tile)
                temp.remove(Tile(tile.suit, tile.number+1))
                temp.remove(Tile(tile.suit, tile.number+2))
        new_sets += _three_sets(temp)

        # Check success
        if len(new_sets) == len(hand)/3:
            return new_sets

    return []

def classify_meld(meld):
    # Types 0- pong, 1- kong, 2- chow, 3- eyes
    meld.sort()
    if len(meld) > 3:
        type = 1
    elif len(meld) < 3:
        type = 3
    elif meld[1] == meld[2]:
        type = 0
    else:
        type = 2
    first_tile = meld[0]

    return (type, first_tile)

def score(hand, melds, round_wind, hand_wind):
    # TODO Implement full rules including winds
    sets = partition_tiles(hand)
    if not sets:
        # Hand can not be made into sets and eyes
        return 0
    
    all_sets = sets + melds
    # Contains type, lowest tile tuples
    # Types 0- pong, 1- kong, 2- chow, 3- eyes
    classified_all_sets = []
    for set in all_sets:
        classified_all_sets.append(classify_meld(set))

    #for set in classified_all_sets:
    #    print(set)
    
    # Set any win as 8
    score = 8

    return score

class Environment:

    def __init__(self):

        self.wall = []
        self.discards = []
        self.hands = [[], [], [], []]
        self.melds = [[], [], [], []]
        self.current_player = 0
        self.last_discarding_player = -1
        self.wining_player = -1
        self.round_wind = 0
        self.hand_wind = 0

    def reset(self):
        # Clear the environment for a new game
        self.wall = create_tiles()
        # Deal 13 tiles to each player
        for hand in self.hands:
            del hand[:]
            for _ in range(13):
                hand.append(self.wall.pop())
        # Reset shown tiles and player
        self.discards = []
        self.melds = [[], [], [], []]
        self.current_player = 0

        # Rotate Winds
        if self.hand_wind == 3:
            self.round_wind = (self.round_wind + 1) % 4
        self.hand_wind = (self.hand_wind + 1) % 4

    def encode_state():

        return

    def print_state(self, player = -1):
        # Prints the game state, includes info for all players unless specified
        # Print wall
        print("Tiles in Wall: " + str(len(self.wall)))
        if self.wall: 
            print("Next Tile in Wall: " + str(self.wall[-1]))
        # Print discards
        print("Tiles in Discards: " + str(len(self.discards)))
        if self.discards: 
            print("Last Discard: " + str(self.discards[-1]))
        # Print Melds
        for p in range(4):
            print("Player " + str(p) + " Melds: ")
            for meld in self.melds[p]:
                for tile in meld:
                    print(tile, end="  ")
            print()
        # Print tiles for a specific player
        if player >= 0:
            print("Player " + str(player) + " Hand: ")
            for tile in self.hands[player]:
                print(tile, end="  ")
            print()
        # Print information for all players
        else:
            for p in range(4):
                print("Player " + str(p) + " Hand: ")
                for tile in self.hands[p]:
                    print(tile, end="  ")
                print()

    def calculate_rewards(self, status, score = 0, discarder = None):
        # Status 0- no win, 1- self draw, 2- non-self
        match status:
            case 0:
                # No more tiles in the wall, punish all players slightly
                return [-0.1,-0.1,-0.1,-0.1]    
            case 1:
                # Current player wins from wall, reward current player double, punish others
                rewards = [-0.5,-0.5,-0.5,-0.5]
                rewards[self.current_player] = score / 5
                return rewards
            case 2:
                # Current player wins from discard, reward current player, punish others, mainly discarding
                rewards = [-0.25,-0.25,-0.25,-0.25]
                rewards[self.current_player] = score / 10 
                rewards[discarder] = -score / 10 
                return rewards
        return [0,0,0,0]

    def play_turn(self, picked_up = False):
        # The current player plays a turn, returns rewards if the game is over and whether the player has picked up
        # Player choses from options 0-13 Discard, 14-16 Kong, 17-18 19 Win
        if len(self.wall) == 0:
            # Game is over with no winner (winning player remains -1)
            return (False, self.calculate_rewards(0))
        if not picked_up:
            # Pick up a tile from the end of the wall
            self.hands[self.current_player].append(self.wall.pop())
        options = []
        # Player can discard a tile
        for i in range(len(self.hands[self.current_player])):
            options.append(i)
            
        # Player could have 0-3 ways to kong from hand (options 14-16), needs additional tile from wall
        if kong_options(self.hands[self.current_player]) and len(self.wall) > 0:
            for i in range(len((kong_options(self.hands[self.current_player])))):
                options.append(14 + i)
        # Player could kong onto exposed pong (option 17-18), needs additional tile from wall
        if kong_onto_meld_options(self.hands[self.current_player], self.melds[self.current_player]) and len(self.wall) > 0:
            for i in range(len(kong_onto_meld_options(self.hands[self.current_player], self.melds[self.current_player]))):
                options.append(17 + i)
        # Player may be able to win (option 19)
        if score(self.hands[self.current_player], self.melds[self.current_player], self.round_wind, self.hand_wind) >= 8:
            options.append(19)

        # Make decision (0-19 relevant)
        # TODO Implement Agent
        choice = random.choice(options)
        if choice < 14:
            # Discard a tile
            self.discards.append(self.hands[self.current_player].pop(choice))
            return (False, [])
        
        elif choice < 17:
            # Kong
            options = kong_options(self.hands[self.current_player])
            tile = options[choice - 14]
            # Add to melds and remove from hand
            self.melds[self.current_player].append([tile] * 4)
            for i in range(4):
                self.hands[self.current_player].remove(tile)
            # Get tile from start of wall
            self.hands[self.current_player].append(self.wall.pop(0))
            # Player has picked up
            return (True, [])
        
        elif choice < 19:
            # Kong onto exposed pong
            options = kong_onto_meld_options(self.hands[self.current_player], self.melds[self.current_player])
            tile = options[choice - 17]
            # Add to melds and remove from hand
            for meld in self.melds[self.current_player]:
                type, first_tile = classify_meld(meld) 
                # Check if pung of the same tile, then add
                if type == 0 and first_tile == tile:
                    meld.append(tile)
            self.hands[self.current_player].remove(tile)
            # Get tile from start of wall
            self.hands[self.current_player].append(self.wall.pop(0))
            # Player has picked up
            return (True, [])

        elif choice == 19:
            # Win, self pickup
            print("Player " + str(self.current_player) + " wins from wall")
            return (False, self.calculate_rewards(1, score(self.hands[self.current_player], self.melds[self.current_player], self.round_wind, self.hand_wind)))
        
    def post_discard(self):
        # Player has discarded a tile, other players have option to pong, chow, kong, or win
        player_options = [[-1], [-1], [-1], [-1]] # -1 for no action, 19 for win, 20 for pong, 21 for kong, 22-24 for chows
        # Check win off discard
        for player in range(4):
            if score(self.hands[player] + [self.discards[-1]], self.melds[player], self.round_wind, self.hand_wind) >= 8:
                player_options[player] += [19]
        # Check Pong
        for player in range(4):
            if self.hands[player].count(self.discards[-1]) >= 2:
                player_options[player] += [20]
        # Check Kong
        for player in range(4):
            if self.hands[player].count(self.discards[-1]) == 3 and len(self.wall) > 0:
                player_options[player] += [21]
        # Check Chow, only possible for player to the right of discarder
        player = (self.current_player + 1) % 4
        if self.discards[-1].suit < 4:
            # If not dragon or wind look for chow
            suit = self.discards[-1].suit
            number = self.discards[-1].number
            if Tile(suit, number - 2) in self.hands[player] and Tile(suit, number - 1) in self.hands[player]:
                player_options[player] += [22]
            elif Tile(suit, number - 1) in self.hands[player] and Tile(suit, number + 1) in self.hands[player]:
                player_options[player] += [23]
            elif Tile(suit, number + 1) in self.hands[player] and Tile(suit, number + 2) in self.hands[player]:
                player_options[player] += [24]

        # Make decisions -1 or 19-24
        player_choices = [[], [], [], []]
        
        for player in range(4):
            # TODO Implement Agent
            player_choices[player] = random.choice(player_options[player])
            #print("Player " + str(player) + " chose option: " + str(player_choices[player]))
            
        # Implement decisions based on priority (from right of discarder if tie)
        for i in range(1, 5):
            if player_choices[(self.current_player + i) % 4] == 19:
                # Win
                discarder = self.current_player
                self.current_player = (self.current_player + i) % 4
                print("Player " + str(self.current_player) + " wins off discard")
                return (False, self.calculate_rewards(2, score(self.hands[self.current_player], self.melds[self.current_player], self.round_wind, self.hand_wind), discarder))
            
        for i in range(1, 5):
            player = (self.current_player + i) % 4
            if player_choices[player] == 20:
                # Pong, add to melds and remove from hand
                tile = self.discards[-1]
                self.melds[player].append([tile] * 3)
                for _ in range(2):
                    self.hands[player].remove(tile)
                # Update current player
                self.current_player = player
                return (False, [])
            elif player_choices[player] == 21:
                # Kong
                tile = self.discards[-1]
                self.melds[player].append([tile] * 4)
                for _ in range(3):
                    self.hands[player].remove(tile)
                # Get tile from start of wall
                self.hands[player].append(self.wall.pop(0))
                # Update current player
                self.current_player = player
                # Player has picked up tile
                return (True, [])

        player = (self.current_player + 1) % 4
        suit = self.discards[-1].suit
        number = self.discards[-1].number
        if player_choices[player] == 22:
                # Chow with discard as highest
                self.melds[player].append([Tile(suit, number - 2), Tile(suit, number - 1), self.discards.pop()])
                self.hands[player].remove(Tile(suit, number - 2))
                self.hands[player].remove(Tile(suit, number - 1))
        elif player_choices[player] == 23:
                # Chow with discard as middle
                self.melds[player].append([Tile(suit, number - 1), Tile(suit, number + 1), self.discards.pop()])
                self.hands[player].remove(Tile(suit, number - 1))
                self.hands[player].remove(Tile(suit, number + 1))
        elif player_choices[player] == 24:
                # Chow with discard as lowest
                self.melds[player].append([Tile(suit, number + 1), Tile(suit, number + 2), self.discards.pop()])
                self.hands[player].remove(Tile(suit, number + 1))
                self.hands[player].remove(Tile(suit, number + 2))

        # Update current player
        self.current_player = player

        return (False, [])
    
    def play_game(self):
        # Play a full game of mahjong
        self.reset()
        picked_up = False
        while True:
            # Play a turn
            picked_up, rewards = self.play_turn(picked_up)
            # Check if game is over
            if rewards:
                break   
            elif not picked_up:
                # Post discard actions, skip if player has picked up
                picked_up, rewards = self.post_discard()
            # Check again if game is over
            if rewards:
                break   
        return rewards

def simulate_iterations(iterations):
    # Create a new instance of Environment for this process
    env = Environment()
    wins = 0
    total_discards = 0
    for _ in range(iterations):
        rewards = env.play_game()
        if rewards != [-0.1, -0.1, -0.1, -0.1]:
            #env.print_state()
            wins += 1
        total_discards += len(env.discards)
    return wins, total_discards / iterations

def main_multiprocess():
    total_iterations = 100000
    num_processes = 10  # Number of processes to run concurrently
    iterations_per_process = total_iterations // num_processes

    overall_wins = 0
    overall_discards = 0.0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(simulate_iterations, iterations_per_process)
                   for _ in range(num_processes)]
        for future in concurrent.futures.as_completed(futures):
            wins, avg_discards = future.result()
            overall_wins += wins
            overall_discards += avg_discards

    overall_discards /= num_processes  # Average discards across processes
    print("Total games won:", overall_wins)
    print("Average number of discards at game end:", overall_discards)

def main():
    
    env = Environment()
    itterations = 1000
    wins = discards = 0
    for _ in range(itterations):
        if env.play_game() != [-0.1, -0.1, -0.1, -0.1]:
            wins += 1
        discards += len(env.discards)
    discards = discards / itterations
    
    print("Done")
    print("Games won: ")
    print(wins)
    print("Average number of discards at game end: ")
    print(discards)

if __name__ == "__main__":
    #main_multiprocess()
    main()