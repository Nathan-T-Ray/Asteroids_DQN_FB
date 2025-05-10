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

import random
from util.vectorsprites import *
from util import *


class Shooter(VectorSprite):

    def __init__(self, position, heading, pointlist, stage):
        VectorSprite.__init__(self, position, heading, pointlist)
        self.bullets = []
        self.stage = stage

    def fireBullet(self, heading, ttl, velocity, isFriendly):
        if (len(self.bullets) < self.maxBullets):
            position = Vector2d(self.position.x, self.position.y)
            newBullet = Bullet(position, heading, self,
                               ttl, velocity, self.stage, isFriendly)
            self.bullets.append(newBullet)
            self.stage.addSprite(newBullet)
            return True

    def bulletCollision(self, target):
        collisionDetected = False
        for bullet in self.bullets:
            if bullet.ttl > 0 and target.collidesWith(bullet):
                collisionDetected = True
                bullet.ttl = 0

        return collisionDetected

# Bullet class


class Bullet(Point):

    def __init__(self, position, heading, shooter, ttl, velocity, stage, isFriendly):
        Point.__init__(self, position, heading, stage)
        self.shooter = shooter
        self.ttl = ttl
        self.velocity = velocity
        self.isFriendly = isFriendly

    def move(self):
        Point.move(self)
        if (self.ttl <= 0):
            self.shooter.bullets.remove(self)
    
    #Prints a string representation of the bullet
    def __str__(self):
                return f"Bullet(position=({self.position.x:.2f}, {self.position.y:.2f}), heading=({self.heading.x:.2f}, {self.heading.y:.2f}), ttl={self.ttl}, velocity=({self.velocity}, isFriendly={self.isFriendly})"