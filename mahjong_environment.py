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

""" Helper functions for scoring standard wins """

def partition_groups(tiles):
    """
    Recursively partitions a sorted list of tiles into groups of 3.
    Each group must be either:
      - a pung (three identical tiles), or
      - a chow (three consecutive numbers in the same suit; honors cannot form chows).
    Returns a list of partitions, where each partition is a list of groups (each group is a list of tiles).
    """
    if not tiles:
        return [[]]
    
    partitions = []
    first = tiles[0]
    
    # Option 1: Try a pung (three identical tiles)
    if tiles.count(first) >= 3:
        new_tiles = list(tiles)
        for _ in range(3):
            new_tiles.remove(first)
        for sub in partition_groups(new_tiles):
            partitions.append([[first, first, first]] + sub)
            
    # Option 2: Try a chow (only for suited tiles that are not honors, and where first.number <= 7)
    if first.suit in {"bamboo", "character", "dot"} and first.number <= 7:
        needed1 = Tile(first.suit, first.number + 1)
        needed2 = Tile(first.suit, first.number + 2)
        if needed1 in tiles and needed2 in tiles:
            new_tiles = list(tiles)
            new_tiles.remove(first)
            new_tiles.remove(needed1)
            new_tiles.remove(needed2)
            for sub in partition_groups(new_tiles):
                partitions.append([[first, needed1, needed2]] + sub)
    return partitions

def classify_group(group):
    """
    Given a group (list of Tile objects), classify it as one of:
      - ("chow", suit, (n, n+1, n+2))
      - ("pung", suit, number)
      - ("kong", suit, number)
    Returns None if the group is invalid.
    """
    if len(group) == 3:
        if group[0].suit == group[1].suit == group[2].suit:
            # Check if it is a pung (all tiles identical)
            if group[0] == group[1] and group[1] == group[2]:
                return ("pung", group[0].suit, group[0].number)
            else:
                nums = sorted([t.number for t in group])
                if nums[0] + 1 == nums[1] and nums[1] + 1 == nums[2]:
                    return ("chow", group[0].suit, tuple(nums))
    elif len(group) == 4:
        # Check for kong (four identical tiles)
        if (group[0].suit == group[1].suit == group[2].suit == group[3].suit and
            group[0] == group[1] == group[2] == group[3]):
            return ("kong", group[0].suit, group[0].number)
    return None

def score_pure_double_chow(groups):
    """
    For each chow that appears at least twice in the same suit with the identical sequence,
    add one point per pair.
    """
    chow_counts = {}
    for g in groups:
        if g[0] == "chow":
            key = (g[1], g[2])  # (suit, sequence)
            chow_counts[key] = chow_counts.get(key, 0) + 1
    score = 0
    for key, count in chow_counts.items():
        score += count // 2
    return score

def score_mixed_double_chow(groups):
    """
    For each chow sequence (ignoring suit) that appears in at least two different suits,
    add one point (provided no pure double chow exists for that sequence).
    """
    seq_suits = {}
    for g in groups:
        if g[0] == "chow":
            seq = g[2]  # the numerical sequence
            seq_suits.setdefault(seq, set()).add(g[1])
    score = 0
    for seq, suits in seq_suits.items():
        if len(suits) >= 2:
            # Only score mixed double if no pure double exists for that sequence.
            pure_exists = any(sum(1 for g in groups if g[0]=="chow" and g[1]==s and g[2]==seq) >= 2 
                             for s in suits)
            if not pure_exists:
                score += 1
    return score

def score_short_straight(groups):
    """
    For each suit, if there are two chow groups whose starting numbers are consecutive 
    (i.e. one chow is, say, 1-2-3 and another is 4-5-6), add one point.
    """
    suit_starts = {}
    for g in groups:
        if g[0] == "chow":
            suit = g[1]
            start = g[2][0]
            suit_starts.setdefault(suit, []).append(start)
    score = 0
    for suit, starts in suit_starts.items():
        starts = sorted(starts)
        for i in range(len(starts) - 1):
            if starts[i+1] == starts[i] + 3:
                score += 1
    return score

def score_two_terminal_chows(groups):
    # TODO needs fixing
    """
    For each suit (bamboo, character, dot), if both 1‑2‑3 and 7‑8‑9 chows are present,
    add one point.
    """
    score = 0
    for suit in {"bamboo", "character", "dot"}:
        has_low = any(g for g in groups if g[0]=="chow" and g[1]==suit and g[2]==(1,2,3))
        has_high = any(g for g in groups if g[0]=="chow" and g[1]==suit and g[2]==(7,8,9))
        if has_low and has_high:
            score += 1
    return score

def score_pung_terminals_honors(groups):
    # TODO needs fixing
    """
    For every pung group (three identical tiles) where the tile is a terminal (1 or 9)
    or is an honor, add one point.
    """
    score = 0
    for g in groups:
        if g[0] == "pung":
            if g[1] == "honor" or g[2] in (1, 9):
                score += 1
    return score

def score_melded_kong(melds):
    """
    For every exposed meld that is a kong (four identical tiles), add one point.
    """
    score = 0
    for meld in melds:
        if len(meld) == 4:
            descriptor = classify_group(meld)
            if descriptor and descriptor[0] == "kong":
                score += 1
    return score

def score_one_voided_suit(tiles):
    # TODO needs fixing
    """
    If the complete hand (tiles from melds and concealed hand) is missing one of the three suits 
    (bamboo, character, or dot), add one point.
    """
    suits = set(t.suit for t in tiles if t.suit in {"bamboo", "character", "dot"})
    return 1 if len(suits) <= 2 else 0

def score_no_honors(tiles):
    # TODO needs fixing
    """
    If the complete hand contains no honor tiles, add one point.
    """
    return 1 if not any(t.suit == "honor" for t in tiles) else 0

def score_dragon_pung(groups):
    # TODO needs fixing
    """
    (Rule 14, 2 points)
    Score 2 points for each pung or kong of Dragon tiles.
    Assumes Dragon tiles have tile.suit == "dragon".
    """
    score = 0
    for g in groups:
        if g and g[0] in ("pung", "kong") and g[1] == "dragon":
            score += 2
    return score

def score_seat_wind(groups, seat_wind):
    # TODO needs fixing
    """
    (Rule 16, 2 points)
    Score 2 points for each pung or kong of the player's seat wind.
    """
    score = 0
    for g in groups:
        if g and g[1] == "wind" and g[2] == seat_wind and g[0] in ("pung", "kong"):
            score += 2
    return score

def score_prevalent_wind(groups, table_wind):
    # TODO needs fixing
    """
    (Rule 15, 2 points)
    Score 2 points for each pung of the table wind.
    Assumes Wind tiles have suit "wind" and tile.number equals the wind value.
    """
    score = 0
    for g in groups:
        if g and g[0] == "pung" and g[1] == "wind" and g[2] == table_wind:
            score += 2
    return score

def score_concealed_hand_bonus(melds, winning_tile_source):
    """
    (Rule 17, 2 points)
    Score 2 points if the hand is completely concealed (no exposed melds)
    and the winning tile was taken from a discard.
    """
    return 2 if len(melds) == 0 and winning_tile_source == "discard" else 0

def score_all_chows(groups):
    """
    (Rule 18, 2 points)
    Score 2 points if every group in the hand is a chow.
    """
    if all(g and g[0] == "chow" for g in groups):
        return 2
    return 0

def score_tile_hog(tiles, groups):
    """
    (Rule 19, 2 points)
    Score 2 points for each tile of a suited type that appears exactly 4 times in the complete hand,
    provided that these four tiles are not declared as a kong in any group.
    """
    from collections import Counter
    counter = Counter(tiles)
    score = 0
    for tile, cnt in counter.items():
        if cnt == 4:
            if not any(g and g[0] == "kong" and g[1] == tile.suit and g[2] == tile.number for g in groups):
                score += 2
    return score

def score_double_pung(groups):
    """
    (Rule 20, 2 points)
    Score 2 points for each distinct number for which there are pungs in two different suits.
    """
    pung_map = {}
    for g in groups:
        if g and g[0] == "pung":
            pung_map.setdefault(g[2], set()).add(g[1])
    score = 0
    for number, suits in pung_map.items():
        if len(suits) >= 2:
            score += 2
    return score

def score_two_concealed_pungs(num_concealed_pungs):
    """
    (Rule 21, 2 points)
    Score 2 points if there are at least two concealed pungs.
    """
    return 2 if num_concealed_pungs >= 2 else 0

def score_all_simples(tiles):
    """
    (Rule 23, 2 points)
    Score 2 points if the complete hand is formed only of simples,
    meaning it contains no terminal (1 or 9) or honor tiles.
    (Assumes suited tiles only for simples are in bamboo, character, or dot.)
    """
    for t in tiles:
        if t.suit not in {"bamboo", "character", "dot"}:
            return 0
        if t.number in (1, 9):
            return 0
    return 2

def is_outside_set(group):
    """
    Returns True if the given group (a list of Tile objects) contains at least one terminal (1 or 9 in suited tiles)
    or an honor tile (any tile not in {"bamboo", "character", "dot"}).
    """
    for tile in group:
        if tile.suit in {"bamboo", "character", "dot"}:
            if tile.number in (1, 9):
                return True
        else:
            return True
    return False

def score_outside_hand(concealed_groups, pair, melds):
    """
    (Rule 24, 4 points)
    Score 4 points if every set of the complete hand—including each concealed group, the pair, and every exposed meld—
    contains at least one terminal or honor tile.
    """
    for group in concealed_groups:
        if not is_outside_set(group):
            return 0
    if not is_outside_set(pair):
        return 0
    for meld in melds:
        if not is_outside_set(meld):
            return 0
    return 4

def score_fully_concealed_hand(melds, winning_tile_source):
    """
    (Rule 25, 4 points)
    Score 4 points if the hand is completely concealed (no exposed melds)
    and the winning tile was drawn from the wall.
    """
    return 4 if len(melds) == 0 and winning_tile_source == "wall" else 0

def score_two_melded_kongs(melds):
    """
    (Rule 26, 4 points)
    Score 4 points if the hand contains at least two claimed (exposed) kongs.
    """
    count = 0
    for meld in melds:
        desc = classify_group(meld)
        if desc and desc[0] == "kong":
            count += 1
    return 4 if count >= 2 else 0

def score_last_tile(winning_tile, discarded_tiles, melds, winning_tile_source):
    # TODO needs fixing
    """
    (Rule 27, 4 points)
    Score 4 points if the winning tile is the last tile of its kind.
    That is, aside from the winning tile, the other three copies of that tile are either in the discard pile
    or used in claimed sets (melds). If the win resulted from robbing a kong (indicated by winning_tile_source),
    no bonus is awarded.
    """
    if winning_tile_source == "kong_rob":
        return 0
    count = discarded_tiles.count(winning_tile)
    for meld in melds:
        count += sum(1 for t in meld if t == winning_tile)
    return 4 if count == 3 else 0

def partition_groups(tiles):
    """
    Recursively partitions a sorted list of tiles into groups.
    Each group may be:
      - a concealed kong (four identical tiles), or
      - a pung (three identical tiles), or 
      - a chow (three consecutive numbers in the same suit; honors cannot form chows).
    """
    if not tiles:
        return [[]]
    
    partitions = []
    first = tiles[0]
    
    # Option 1: Try a pung (group of 3 identical tiles)
    if tiles.count(first) >= 3:
        new_tiles = list(tiles)
        for _ in range(3):
            new_tiles.remove(first)
        for sub in partition_groups(new_tiles):
            partitions.append([[first, first, first]] + sub)
            
    # Option 2: Try a chow (only for suited tiles that are not honors, and where first.number <= 7)
    if first.suit in {"bamboo", "character", "dot"} and first.number <= 7:
        needed1 = Tile(first.suit, first.number + 1)
        needed2 = Tile(first.suit, first.number + 2)
        if needed1 in tiles and needed2 in tiles:
            new_tiles = list(tiles)
            new_tiles.remove(first)
            new_tiles.remove(needed1)
            new_tiles.remove(needed2)
            for sub in partition_groups(new_tiles):
                partitions.append([[first, needed1, needed2]] + sub)
    
    return partitions

def classify_group(group):
    """
    Given a group (a list of tile objects), classify it as one of:
      - ("chow", suit, (a, a+1, a+2))
      - ("pung", suit, number)
      - ("kong", suit, number)
    Returns None if the group is invalid.
    """
    if len(group) == 3:
        # All tiles must have the same suit.
        if group[0].suit == group[1].suit == group[2].suit:
            # Check for pung (three identical tiles)
            if group[0] == group[1] and group[1] == group[2]:
                return ("pung", group[0].suit, group[0].number)
            else:
                # Check for chow (three consecutive numbers)
                nums = sorted([t.number for t in group])
                if nums[0] + 1 == nums[1] and nums[1] + 1 == nums[2]:
                    return ("chow", group[0].suit, tuple(nums))
    elif len(group) == 4:
        # Check for kong (four identical tiles)
        if (group[0].suit == group[1].suit == group[2].suit == group[3].suit and
            group[0] == group[1] == group[2] == group[3]):
            return ("kong", group[0].suit, group[0].number)
    return None

""" Scoring functions for different types of winning hands """

def score_standard_win(melds, hand, winning_tile_source, table_wind, seat_wind, last_tile, extra_points):
    # TODO handle wait patterns (9,10,11) , conceled kongs (22)
    """
    Score a standard winning mahjong hand.

    Parameters:
      - melds: a list of exposed melds (each meld is a list of Tile objects forming a set).
      - hand: a list of concealed Tile objects.
      - winning_tile_source: a string indicating the source of the winning tile.
          If "wall", the winning tile was drawn from the wall (self-drawn);
          if "discard", it was picked up from another player's discard.
      - table_wind: the table (round’s) wind tile (e.g. "east", "south", etc.)
      - seat_wind: the player's seat wind (e.g. "east", "south", etc.)

    (Note: Wait pattern bonuses are omitted here because no wait type is provided.)
    """
    # Validate the concealed hand tile count.
    num_exposed_groups = len(melds)  # Each meld counts as one group.
    concealed_needed_groups = 4 - num_exposed_groups
    expected_concealed_tiles = 3 * concealed_needed_groups + 2
    if len(hand) != expected_concealed_tiles:
        raise ValueError(f"Concealed hand should have {expected_concealed_tiles} tiles, got {len(hand)}.")
    
    partitions = partition_concealed_hand(hand)
    if not partitions:
        raise ValueError("No valid partition found for the concealed hand.")
    
    best_score = -float('inf')
    for part in partitions:
        concealed_groups = part['groups']
        pair = part['pair']  # The pair is available for wait patterns if needed.
        
        # Keep concealed and meld groups separate for later bonuses.
        concealed_descriptors = [classify_group(g) for g in concealed_groups]
        meld_descriptors = [classify_group(m) for m in melds]
        all_groups = meld_descriptors + concealed_descriptors
        
        # 1 point rules
        score = 0
        score += score_pure_double_chow(all_groups)
        score += score_mixed_double_chow(all_groups)
        score += score_short_straight(all_groups)
        score += score_two_terminal_chows(all_groups)
        score += score_pung_terminals_honors(all_groups)
        # Score exposed melds for melded kong.
        score += score_melded_kong(melds)
        
        # Score whole-hand patterns.
        full_tiles = []
        for meld in melds:
            full_tiles.extend(meld)
        full_tiles.extend(hand)
        score += score_one_voided_suit(full_tiles)
        score += score_no_honors(full_tiles)
        
        # Score self-drawn win.
        if winning_tile_source == "wall":
            score += 1

        # 2 point rules
        score += score_dragon_pung(all_groups)
        score += score_prevalent_wind(all_groups, table_wind)
        score += score_seat_wind(all_groups, seat_wind)
        score += score_concealed_hand_bonus(melds, winning_tile_source)
        score += score_all_chows(all_groups)
        score += score_tile_hog(full_tiles, all_groups)
        score += score_double_pung(all_groups)

        # Count concealed pungs (only those from concealed groups)
        num_concealed_pungs = sum(1 for desc in concealed_descriptors if desc and desc[0] == "pung")
        score += score_two_concealed_pungs(num_concealed_pungs)
        score += score_all_simples(full_tiles)

        # 4 point rules
        score += score_outside_hand(concealed_groups, pair, melds)
        score += score_fully_concealed_hand(melds, winning_tile_source)
        score += score_two_melded_kongs(melds)
        score += score_last_tile(winning_tile, discarded_tiles, melds, winning_tile_source)
        
        best_score = max(best_score, score)
    
    return best_score


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

    