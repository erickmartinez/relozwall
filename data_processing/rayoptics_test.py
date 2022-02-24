import wget
# initialization
from rayoptics.environment import *

url = 'http://www.futurecrew.com/skaven/song_files/mp3/razorback.mp3'
filename = wget.download(url)