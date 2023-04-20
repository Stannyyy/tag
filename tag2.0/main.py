from moderator import Moderator
from player import Player

p1 = Player('pietje')
p2 = Player('jantje')
test = Moderator([p1, p2])
test.play()