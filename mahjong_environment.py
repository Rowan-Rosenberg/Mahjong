"""
    Environment to simulate a mahjong competion rules (official Chineese version) game for machine learning
    Written by Rowan Rosenberg February 2025
"""

from enum import Enum
import random

""" Tiles """

class Tile():
    # Type 0-Bamboo, 1-Character, 2-Dot, 3-Wind, 4-Dragon
    # Number 1-9 for regular tile, Winds 1-4 east, south, west, north, Dragons 1-3 red, green, white
    def __init__(self, type, number):
        self.type = type
        self.number = number

        # ONLY USE FOR DEBUGGING
        self.name = TileType(type).name + "-" + str(number)

    def __eq__(self,other):
        return (self.type == other.type) and (self.number == other.number)

    # Returns Tuple form of tile
    def key(self): 
        return (self.type, self.number)

# ONLY USED FOR DEBUGGING (PRINTING TO CONSOLE)
class TileType(Enum):
    BAMBOO = 0
    CHARACTER = 1
    DOT = 2
    WIND = 3
    DRAGON = 4


""" Functions to check for winning hands """

def is_standard_win(tiles):
    
    if len(tiles) != 14:
        return False
    
    # Check if the hand has 4 melds and a pair
    # Build a dictionary counting the tiles.
    # The key is a tuple (suit, number) and the value is the count.
    counts = {}
    for tile in tiles:
        key = tile.key()
        counts[key] = counts.get(key, 0) + 1

    # Try every possible pair candidate.
    for key, count in list(counts.items()):
        if count >= 2:
            # Remove the pair from a copy of the counts.
            counts_copy = counts.copy()
            counts_copy[key] -= 2
            if counts_copy[key] == 0:
              del counts_copy[key]

            # Check if the remaining tiles can be arranged into 4 melds.
            if remove_melds(counts_copy):
                return True

        return False

def remove_melds(counts):
        
    # Recursively attempts to remove melds (pungs or chows) from the counts dictionary.
    # Returns True if all tiles can be grouped into melds; otherwise, returns False.
        
    # If no tiles are left, we have successfully formed melds.
    if not counts:
        return True

    # Pick the lowest tile (using tuple ordering).
    key = min(counts)
    suit, number = key

    # Option 1: Try a pung (three identical tiles).
    if counts.get(key, 0) >= 3:
        counts_copy = counts.copy()
        counts_copy[key] -= 3
        if counts_copy[key] == 0:
            del counts_copy[key]
        if remove_melds(counts_copy):
            return True

    # Option 2: Try a chow (a sequence of three consecutive numbers in the same suit).
    # Note: Suits 3 and 4 are not allowed to form consecutive sets.
    if suit not in (3, 4):
        key1 = (suit, number + 1)
        key2 = (suit, number + 2)
        if counts.get(key1, 0) > 0 and counts.get(key2, 0) > 0:
            counts_copy = counts.copy()
            # Remove one instance each of the consecutive tiles.
            counts_copy[key] -= 1
            if counts_copy[key] == 0:
                del counts_copy[key]

            counts_copy[key1] -= 1
            if counts_copy[key1] == 0:
                del counts_copy[key1]

            counts_copy[key2] -= 1
            if counts_copy[key2] == 0:
                del counts_copy[key2]

            if remove_melds(counts_copy):
                return True

    # If neither a pung nor a chow could be removed, this path fails.
    return False

def is_seven_pairs_win(hand):
        
    # Check if the hand is made of 7 pairs.
    # If a tile appears 4 times, it counts as two pairs.
        
    if len(hand) != 14:
        return False
 
    counts = {}
    for tile in hand:
        key = tile.key()
        counts[key] = counts.get(key, 0) + 1

    total_pairs = 0
    for count in counts.values():
        # Every tile count must be even for the hand to be made solely of pairs.
        if count % 2 != 0:
            return False
        total_pairs += count // 2

    return total_pairs == 7

def is_thirteen_orphans_win(hand):
        
    # Terminals (1 and 9) from each numbered suit (suits 0, 1, and 2)
    # All honor tiles from suit 3 and 4
    
    if len(hand) != 14:
        return False

    # Define the set of orphan tile keys.
    orphan_keys = set()
    # Terminals from numbered suits
    for suit in (0, 1, 2):
        orphan_keys.add((suit, 1))
        orphan_keys.add((suit, 9))
    # Winds
    for number in range(1, 5):
        orphan_keys.add((3, number))
    # Dragons
    for number in range(1, 4):
        orphan_keys.add((4, number))

    counts = {}
    for tile in hand:
        key = tile.key()
        counts[key] = counts.get(key, 0) + 1

    # All tiles in the hand must be in the orphan set.
    for key in counts:
        if key not in orphan_keys:
            return False

    # Each orphan key must appear at least once.
    for key in orphan_keys:
        if key not in counts:
            return False

    # There must be exactly one duplicate among the orphan tiles (forming the pair).
    pair_found = False
    for key in orphan_keys:
        c = counts.get(key, 0)
        if c == 1:
            continue
        elif c == 2:
            if pair_found:
                # More than one pair found
                return False
            pair_found = True
        else:
            # Any count greater than 2 is invalid.
            return False

    return pair_found

""" Scoring functions for different types of winning hands """

def score_standard_win(melds, hand):
    # Calculate the score for a 4 melds and a pair hand
    return 10

def score_seven_pairs(hand):
    # Calculate the score for a 7 pairs hand
    return 40

def score_thirteen_orphans():
    # Calculate the score for a 13 orphans hand
    return 50

""" Environment class """

class MahjongEnvironment:

    # Create new object
    def __init__(self):
        self.wall = []
        self.discard_pile = []
        self.player_hands = [[] for _ in range(4)]
        self.player_melds = [[] for _ in range(4)]
        self.current_player = 0
        self.discarding_player = 0
        self.has_picked_up = False
        self.reset()

    # Create fresh game environment
    def reset(self):
        
        # Clear the wall, melds, hands and discard pile
        self.wall = []
        self.discard_pile = []
        self.player_hands = [[] for _ in range(4)]
        self.player_melds = [[] for _ in range(4)]
        self.current_player = 0
        self.discarding_player = 0
        # Keep track of whether the current player has picked up a tile
        self.has_picked_up = False

        # Add the regular tiles to the wall
        for i in range(3):
            for j in range(1, 10):
                self.wall += [Tile(i, j)] * 4

        # Add the wind tiles to the wall
        for i in range(1, 5):
            self.wall += [Tile(3, i)] * 4

        # Add the dragon tiles to the wall
        for i in range(1, 4):
            self.wall += [Tile(4, i)] * 4

        # Shuffle the tiles
        random.shuffle(self.wall)
        
        # Deal the tiles to the players
        for i in range(4):
            for _ in range(13):
                self.player_hands[i].append(self.wall.pop())

    """ Rewards calculation and scoring """

    def calculate_rewards(self, situation, score = None):
        
        match situation:
            case 0:
                # No more tiles in the wall, punish all players slightly
                return [-0.1,-0.1,-0.1,-0.1]    
            case 1:
                # Current player wins from wall, reward current player, punish others
                rewards = [-0.5,-0.5,-0.5,-0.5]
                rewards[self.current_player] = score / 5  # Reward winning player, doubled for winning from wall
                return rewards
            case 2:
                # Current player wins from discard, reward current player, punish others, mainly discaarding
                rewards = [-0.1,-0.1,-0.1,-0.1]
                rewards[self.current_player] = score / 10   # Reward winning player
                rewards[self.discarding_player] = -score / 10  # Punish discarding player
                return rewards
            case _:
                # Default return
                return [0,0,0,0]

    def score(self, hand, melds):
    
        # Get the player's hand and melds
        tiles = hand + melds

        # Score remains 0 if the hand is not winning
        score = 0

        # Possible wins: 4 melds and a pair, 7 pairs, or 13 orphans
        if is_standard_win(tiles):
            score = score_standard_win(melds, hand)
        
        if is_seven_pairs_win(tiles):
            score = score_seven_pairs(hand)

        if is_thirteen_orphans_win(tiles):
            score = score_thirteen_orphans()
    
        return score
    
    """ Interupt functions"""

    def can_pung(tile, hand):
        # Check if a tile can be used to form a pung
        return hand.count(tile) >= 2 

    def chow_options(tile, hand):
        # Check if a tile can be used to form a chow
        options = []
        if tile.type > 2:
            # Winds and dragons cannot be used in chows
            return []
        # Check if two preceeding tiles are in the hand
        if hand.count(Tile(tile.type, tile.number - 1)) > 0 and hand.count(Tile(tile.type, tile.number - 2)) > 0:
            options.append(1)
        # Check if a preceeding and following tile are in the hand
        if hand.count(Tile(tile.type, tile.number - 1)) > 0 and hand.count(Tile(tile.type, tile.number + 1)) > 0:
            options.append(2)
        # Check if two following tiles are in the hand
        if hand.count(Tile(tile.type, tile.number + 1)) > 0 and hand.count(Tile(tile.type, tile.number + 2)) > 0:
            options.append(3)
        
        return options

    def can_kong(tile, hand):
        # Check if a tile can be picked up to form a kong
        if hand.count(tile) == 3:
            return True
        else:
            return False

    def kong_from_hand_options(hand):
        # Check if a player can kong from their hand, returns the options
        options = []
        hand
        for tile in hand:
            if hand.count(tile) == 4 and tile not in options:
                options.append(tile)
        return options

    """ Game cycle """

    # Pick up and discard cycle, returns rewards and whether the game is over
    # Decision 0-13: Discard tile at index, 14 declare win
    def play_turn(self):

        hand = self.player_hands[self.current_player]
        melds = self.player_melds[self.current_player]

        # Check whether a tile must be picked up
        if not self.has_picked_up:

            # Check if the wall is empty, if so the game is over
            if len(self.wall) == 0:

                rewards = self.calculate_rewards(0)
                return rewards, True # Game is over, assign rewards

            else:
                # Pick up a tile from the end of the wall
                self.player_hands[self.current_player].append(self.wall.pop(-1))
                self.has_picked_up = True

        # Check if the player can kong from their hand
        kong_tiles = self.kong_from_hand_options(hand)
        if kong_tiles:
            # Decide whether to kong, and which tile to kong
            options = [i for i in range(len(kong_tiles) + 1)]

            # TODO SWAP WITH AGENTS DECISION MAKING
            decision = random.choice(options)

            if decision == 0:
                # Don't kong
                pass
            else:
                # Kong
                tile = kong_tiles[decision - 1]
                count = 4
                for tile in self.player_hands[self.current_player]:
                    # Remove 4 matches from hand
                    if tile == tile and count > 0:
                        self.player_hands[self.current_player].pop(self.player_hands[self.current_player].index(tile))
                        count -= 1
                # Create meld with 4 copies
                self.player_melds[self.current_player] += [tile] * 4
                # Get tile from other end of wall
                self.player_hands[self.current_player].append(self.wall.pop(0))
                # Continue from konging player
                self.has_picked_up = True
                return [0,0,0,0], False

            return [0,0,0,0], False

        # Can discard any tile in hand
        possible_actions = [i for i in range(14)]

        # Check is hand is winning
        if self.score(hand,melds) > 0:
            possible_actions.append(14)
        
        # TODO SWAP WITH AGENTS DECISION MAKING
        decision = random.choice(possible_actions)

        # Winning decision
        if decision == 14:
            return self.calculate_rewards(1, self.score(hand,melds)), True # Game is won, assign rewards

        # Discard decision
        tile = self.player_hands[self.current_player].pop(decision)
        self.discard_pile.append(tile)
        self.has_picked_up = False
        self.discarding_player = self.current_player

        """ Post Discard """

        discard = self.discard_pile[-1]

        # Wins
        for player in range(4):
            # Check if the tile can be used to win
            if self.score(discard + self.player_hands[player], self.player_melds[player]):
                # Decide whether to declare win
                # TODO SWAP WITH AGENTS DECISION MAKING
                declare = random.choice([True,False])
                if declare:
                    # Handle win off discard
                    rewards = self.calculate_rewards(2, self.score(self.player_hands[player], self.player_melds[player]))
                    return rewards, True # Game is won, assign rewards
                
        # Pungs and Kongs
        for player in range(4):

            if self.can_kong(discard,self.player_hands[player]):
                # Decide whether to kong
                # TODO SWAP WITH AGENTS DECISION MAKING
                kong = random.choice([True,False])
                if kong:
                    # Handle kong
                    count = 3
                    for tile in self.player_hands[player]:
                        # Remove 3 matches from hand
                        if tile == discard and count > 0:
                            self.player_hands[player].pop(self.player_hands[player].index(tile))
                            count -= 1
                    # Remove discard from pile
                    tile = self.discard_pile.pop()
                    # Create meld with 4 copies
                    self.player_melds[player] += [tile] * 4
                    # Get tile from other end of wall
                    self.player_hands[player].append(self.wall.pop(0))
                    # Continue from konging player
                    self.current_player = player
                    self.has_picked_up = True

                    return [0,0,0,0], False

            if self.can_pung(discard,self.player_hands[player]):
                # Decide whether to pung
                # TODO SWAP WITH AGENTS DECISION MAKING
                pung = random.choice([True,False])
                if pung:
                    # Handle pung
                    count = 2
                    for tile in self.player_hands[player]:
                        # Remove 2 matches from hand
                        if tile == discard and count > 0:
                            self.player_hands[player].pop(self.player_hands[player].index(tile))
                            count -= 1
                    # Remove discard from pile
                    tile = self.discard_pile.pop()
                    # Create meld with 3 copies
                    self.player_melds[player] += [tile] * 3
                    # Continue from punging player
                    self.current_player = player
                    self.has_picked_up = True

                    return [0,0,0,0], False


        # Chow for player to right
        chow_options = self.chow_options(discard, self.player_hands[(self.current_player + 1) % 4])
        # Check if list is empty
        if chow_options:
            # Update player to right
            self.current_player = (self.current_player + 1) % 4
            # Choose an option, or none
            chow_options.append(0)

            # TODO SWAP WITH AGENTS DECISION MAKING
            choice = random.choice(chow_options)

            match choice:
                case 0:
                    # Do nothing if choice is 0
                    pass
                case 1:
                    # Chow with discard as highest
                    tile_1_index = self.player_hands[self.current_player].index(Tile(discard.type, discard.number -2))
                    tile_2_index = self.player_hands[self.current_player].index(Tile(discard.type, discard.number -1))
                    # Remove from Hand and add to melds
                    self.player_melds[self.current_player].append(self.player_hands[self.current_player].pop(tile_1_index))
                    self.player_melds[self.current_player].append(self.player_hands[self.current_player].pop(tile_2_index))
                    self.player_melds[self.current_player].append(discard)
                case 2:
                    # Chow with discard as middle
                    tile_1_index = self.player_hands[self.current_player].index(Tile(discard.type, discard.number -1))
                    tile_2_index = self.player_hands[self.current_player].index(Tile(discard.type, discard.number +1))
                    # Remove from Hand and add to melds
                    self.player_melds[self.current_player].append(self.player_hands[self.current_player].pop(tile_1_index))
                    self.player_melds[self.current_player].append(discard)
                    self.player_melds[self.current_player].append(self.player_hands[self.current_player].pop(tile_2_index))
                case 3:
                    # Chow with discard as lowest
                    tile_1_index = self.player_hands[self.current_player].index(Tile(discard.type, discard.number +1))
                    tile_2_index = self.player_hands[self.current_player].index(Tile(discard.type, discard.number +2))
                    # Remove from Hand and add to melds
                    self.player_melds[self.current_player].append(discard)
                    self.player_melds[self.current_player].append(self.player_hands[self.current_player].pop(tile_1_index))
                    self.player_melds[self.current_player].append(self.player_hands[self.current_player].pop(tile_2_index))


        # No post discard actions
        self.current_player = (self.current_player + 1) % 4
        self.has_picked_up = False
        return [0,0,0,0], False

    def print_game_state(self):
        print("********* Wall *********")
        for tile in self.wall:
            print(tile.name)

        print("********* Discard *********")
        for tile in self.discard_pile:
            print(tile.name)
        
        # Print player melds
        for meld in self.player_melds:
            print("********* Meld " + str(self.player_melds.index(meld)) + " *********")
            for tile in meld:
                print(tile.name)
        
        # Print player hands
        for hand in self.player_hands:
            print("********* Hand " + str(self.player_hands.index(hand)) + " *********")
            for tile in hand:
                print(tile.name)
        
        # Print current player
        print("********* Current Player *********")
        print(self.current_player)

        return
    

    
""" Debugging """

if __name__ == "__main__":

    env = MahjongEnvironment()

    env.print_game_state()

    