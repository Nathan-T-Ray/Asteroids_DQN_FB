#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright (C) 2008  Nick Redshaw
#    Copyright (C) 2018  Francisco Sanchez Arroyo
#

import pygame
import os

# Initialize the mixer
pygame.mixer.init()

# Construct the relative path
current_dir = os.path.dirname(__file__)
res_dir = os.path.join(current_dir, "../res")

# Load the sounds
sounds = {}
sounds["fire"] = pygame.mixer.Sound(os.path.join(res_dir, "FIRE.WAV"))
sounds["explode1"] = pygame.mixer.Sound(os.path.join(res_dir, "EXPLODE1.WAV"))
sounds["explode2"] = pygame.mixer.Sound(os.path.join(res_dir, "EXPLODE2.WAV"))
sounds["explode3"] = pygame.mixer.Sound(os.path.join(res_dir, "EXPLODE3.WAV"))
sounds["lsaucer"] = pygame.mixer.Sound(os.path.join(res_dir, "LSAUCER.WAV"))
sounds["ssaucer"] = pygame.mixer.Sound(os.path.join(res_dir, "SSAUCER.WAV"))
sounds["thrust"] = pygame.mixer.Sound(os.path.join(res_dir, "THRUST.WAV"))
sounds["sfire"] = pygame.mixer.Sound(os.path.join(res_dir, "SFIRE.WAV"))
sounds["extralife"] = pygame.mixer.Sound(os.path.join(res_dir, "LIFE.WAV"))

def playSound(soundName):
    if soundName in sounds:
        sounds[soundName].play()
    else:
        print(f"Sound '{soundName}' not found.")

def stopSound(soundName):
    if soundName in sounds:
        sounds[soundName].stop()
    else:
        print(f"Sound '{soundName}' not found.")

def playSoundContinuous(soundName):
    if soundName in sounds:
        sounds[soundName].play(-1)
    else:
        print(f"Sound '{soundName}' not found.")