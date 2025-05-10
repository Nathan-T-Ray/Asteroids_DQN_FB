#!/usr/bin/env python3
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
import sys
import os
import random
import math
from pygame.locals import *
from util.vectorsprites import *
from ship import *
from stage import *
from badies import *
from shooter import *
# from soundManager import *
import numpy as np
from q_learning import QLearningAgent
from collections import deque

class Asteroids:

    explodingTtl = 180

    def __init__(self):
        self.stage = Stage('Atari Asteroids', (1024, 768))
        self.paused = False
        self.showingFPS = False
        self.frameAdvance = False
        self.gameState = "attract_mode"
        self.rockList = []
        self.createRocks(3)
        self.saucer = None
        self.secondsCount = 1
        self.score = 0
        self.ship = None
        self.lives = 0
        self.prev_lives = 0
        self.prev_score = 0
        self.prev_state = None
        self.q_agent = QLearningAgent(input_size=209, num_actions=5, learning_rate=0.00005)
        self.rocks_destroyed = 0
        self.shots_fired = 0
        self.game_transitions = []

    def initialiseGame(self):
        self.gameState = 'playing'
        [self.stage.removeSprite(sprite) for sprite in self.rockList]
        if self.saucer is not None:
            self.killSaucer()
        self.startLives = 5
        self.createNewShip()
        self.createLivesList()
        self.score = 0
        self.rockList = []
        self.numRocks = 3
        self.nextLife = 10000
        self.createRocks(self.numRocks)
        self.secondsCount = 1
        self.rocks_destroyed = 0
        self.shots_fired = 0
        self.game_transitions = []

    def createNewShip(self):
        if self.ship:
            [self.stage.spriteList.remove(debris)
             for debris in self.ship.shipDebrisList]
        self.ship = Ship(self.stage)
        self.stage.addSprite(self.ship.thrustJet)
        self.stage.addSprite(self.ship)

    def createLivesList(self):
        self.lives += 1
        self.livesList = []
        for i in range(1, self.startLives):
            self.addLife(i)

    def addLife(self, lifeNumber):
        self.lives += 1
        ship = Ship(self.stage)
        self.stage.addSprite(ship)
        ship.position.x = self.stage.width - \
            (lifeNumber * ship.boundingRect.width) - 10
        ship.position.y = 0 + ship.boundingRect.height
        self.livesList.append(ship)

    def createRocks(self, numRocks):
        for _ in range(0, numRocks):
            position = Vector2d(random.randrange(-10, 10),
                                random.randrange(-10, 10))
            newRock = Rock(self.stage, position, Rock.largeRockType)
            self.stage.addSprite(newRock)
            self.rockList.append(newRock)

    def extract_features(self):
        features = []
        stage_width, stage_height = self.stage.width, self.stage.height
        max_rock_velocity = Rock.velocities[Rock.smallRockType]
        max_saucer_velocity = Saucer.velocities[Saucer.smallSaucerType]
        max_distance = math.hypot(stage_width, stage_height)

        # Ship features: [pos_x, pos_y, head_x, head_y, angle, visible, in_hyperspace, velocity_magnitude, thrust_active, bullet_count]
        if self.ship and self.ship.visible:
            velocity_magnitude = math.hypot(self.ship.heading.x, self.ship.heading.y) / Ship.maxVelocity
            thrust_active = 1.0 if self.ship.thrustJet.accelerating else 0.0
            bullet_count = len(self.ship.bullets) / Ship.maxBullets
            ship_features = [
                self.ship.position.x / stage_width,
                self.ship.position.y / stage_height,
                self.ship.heading.x / Ship.maxVelocity,
                self.ship.heading.y / Ship.maxVelocity,
                self.ship.angle / 360.0,
                1.0,
                1.0 if self.ship.inHyperSpace else 0.0,
                velocity_magnitude,
                thrust_active,
                bullet_count
            ]
        else:
            ship_features = [0.0] * 10
        features.extend(ship_features)

        # Rock features: [pos_x, pos_y, head_x, head_y, rock_type] for up to 32 sorted by proximity
        rock_features = []
        closest_rock_distance = 1.0
        closest_rock_rel_pos = [0.0, 0.0]
        closest_rock_rel_vel = [0.0, 0.0]

        if self.ship and self.ship.visible:
            rocks_with_distance = []
            for rock in self.rockList:
                dx = rock.position.x - self.ship.position.x
                dy = rock.position.y - self.ship.position.y
                dx = min(dx, stage_width - dx, key=abs)
                dy = min(dy, stage_height - dy, key=abs)
                distance = math.hypot(dx, dy)
                rocks_with_distance.append((rock, distance))
            rocks_with_distance.sort(key=lambda x: x[1]) # distance sort
            sorted_rocks = [rock for rock, _ in rocks_with_distance] 

            if sorted_rocks:
                closest_rock = sorted_rocks[0]
                dx = closest_rock.position.x - self.ship.position.x
                dy = closest_rock.position.y - self.ship.position.y
                dx = min(dx, stage_width - dx, key=abs)
                dy = min(dy, stage_height - dy, key=abs)
                closest_rock_distance = math.hypot(dx, dy) / max_distance
                #account 4 screen size, norm
                closest_rock_rel_pos = [dx / stage_width, dy / stage_height]
                dvx = (closest_rock.heading.x - self.ship.heading.x) / max_rock_velocity
                dvy = (closest_rock.heading.y - self.ship.heading.y) / max_rock_velocity
                closest_rock_rel_vel = [dvx, dvy]
        else:
            sorted_rocks = self.rockList

        for rock in sorted_rocks[:32]:
            rock_features.extend([
                rock.position.x / stage_width,
                rock.position.y / stage_height,
                rock.heading.x / max_rock_velocity,
                rock.heading.y / max_rock_velocity,
                rock.rockType / 2.0
            ])
        rock_features.extend([0.0] * (5 * (32 - len(sorted_rocks[:32]))))
        features.extend(rock_features)

        # Saucer features: [pos_x, pos_y, head_x, head_y, saucer_type]
        if self.saucer:
            saucer_features = [
                self.saucer.position.x / stage_width,
                self.saucer.position.y / stage_height,
                self.saucer.heading.x / max_saucer_velocity,
                self.saucer.heading.y / max_saucer_velocity,
                self.saucer.saucerType / 1.0
            ]
        else:
            saucer_features = [0.0] * 5
        features.extend(saucer_features)

        # Bullet features: [pos_x, pos_y, head_x, head_y, is_friendly] for up to 4 enemy bullets
        bullet_features = []
        bullets = []
        if self.saucer:
            bullets.extend([b for b in self.saucer.bullets if not b.isFriendly]) # check if bullets are hostile
            #NOTE: CHECKME for possible malfuction

        closest_bullet_distance = 1.0
        closest_bullet_rel_pos = [0.0, 0.0]
        closest_bullet_rel_vel = [0.0, 0.0]

        if bullets and self.ship and self.ship.visible:
            bullet_distances = []

            for bullet in bullets:
                dx = bullet.position.x - self.ship.position.x
                dy = bullet.position.y - self.ship.position.y
                dx = min(dx, stage_width - dx, key=abs)
                dy = min(dy, stage_height - dy, key=abs)
                distance = math.hypot(dx, dy)
                bullet_distances.append((bullet, distance))
            bullet_distances.sort(key=lambda x: x[1])

            if bullet_distances:
                closest_bullet, distance = bullet_distances[0]
                closest_bullet_distance = distance / max_distance
                dx = closest_bullet.position.x - self.ship.position.x
                dy = closest_bullet.position.y - self.ship.position.y
                dx = min(dx, stage_width - dx, key=abs)
                dy = min(dy, stage_height - dy, key=abs)
                closest_bullet_rel_pos = [dx / stage_width, dy / stage_height]
                dvx = (closest_bullet.heading.x - self.ship.heading.x) / Ship.bulletVelocity
                dvy = (closest_bullet.heading.y - self.ship.heading.y) / Ship.bulletVelocity
                closest_bullet_rel_vel = [dvx, dvy]

        for bullet in bullets[:4]:
            bullet_features.extend([
                bullet.position.x / stage_width,
                bullet.position.y / stage_height,
                bullet.heading.x / Ship.bulletVelocity,
                bullet.heading.y / Ship.bulletVelocity,
                0.0
            ])
        bullet_features.extend([0.0] * (5 * (4 - len(bullets[:4]))))
        features.extend(bullet_features)

        # Game state features: [lives, score, rocks_remaining, saucer_present, closest_rock_distance, closest_bullet_distance]
        game_state_features = [
            self.lives / self.startLives,
            min(self.score / 10000.0, 1.0),  # Cap score normalization
            len(self.rockList) / 32.0,
            1.0 if self.saucer else 0.0,
            closest_rock_distance,
            closest_bullet_distance
        ]
        features.extend(game_state_features)

        # Relative features: [closest_rock_dx, closest_rock_dy, closest_rock_dvx, closest_rock_dvy, closest_bullet_dx, closest_bullet_dy, closest_bullet_dvx, closest_bullet_dvy]
        relative_features = closest_rock_rel_pos + closest_rock_rel_vel + closest_bullet_rel_pos + closest_bullet_rel_vel
        features.extend(relative_features)

        # Total features: 10 (ship) + 160 (32 rocks * 5) + 5 (saucer) + 20 (4 bullets * 5) + 6 (game state) + 8 (relative) = 209
        return np.array(features, dtype=np.float32)
    
    #NOTE: REWARDS

    def compute_reward(self, curr_state, action_idx):
        reward = 0.0
        max_distance = math.hypot(self.stage.width, self.stage.height)

        reward += (self.score - self.prev_score) * 0.1  # Scale score to avoid dominance

        if self.lives < self.prev_lives:
            reward -= 50.0

        reward += 0.01

        # NOTE: Proximity malus. ADJUSTME!
        closest_rock_distance = curr_state[-8] * max_distance  # closest_rock_distance (normalized)
        if closest_rock_distance < 100.0:
            reward -= (100.0 - closest_rock_distance) * 0.05

        closest_bullet_distance = curr_state[-7] * max_distance  # closest_bullet_distance (normalized)
        if closest_bullet_distance < 50.0:
            reward -= (50.0 - closest_bullet_distance) * 0.1

        # NOTE: Firing reward. ADJUSTME!
        if action_idx == 4 and closest_rock_distance < 200.0:
            reward += 0.5

        return reward

    def playGame(self):
        clock = pygame.time.Clock()
        frameCount = 0.0
        timePassed = 0.0
        self.fps = 0.0
        action_idx = 0

        try:
            while True:
                timePassed += clock.tick(0) 

                # NOTE: can set to zero or no val for overdrive speed
                #might be screwing with the machine though ADJUSTME

                frameCount += 1
                if frameCount % 10 == 0:
                    self.fps = round((frameCount / (timePassed / 1000.0)))
                    timePassed = 0
                    frameCount = 0

                self.secondsCount += 1
                self.input(pygame.event.get())
                if self.paused and not self.frameAdvance:
                    self.displayPaused()
                    continue

                self.stage.screen.fill((10, 10, 10))
                self.stage.moveSprites()
                self.stage.drawSprites()
                self.doSaucerLogic()
                self.displayScore()
                self.displayEpsilon()
                if self.showingFPS:
                    self.displayFps()
                self.checkScore()

                if self.gameState == 'playing':
                    curr_state = self.extract_features()
                    action_idx = self.q_agent.get_action(curr_state)
                    self.apply_ai_action(action_idx)

                    reward = self.compute_reward(curr_state, action_idx)
                    if self.prev_state is not None:
                        transition = (self.prev_state, action_idx, reward, curr_state)
                        self.q_agent.train([transition])
                        self.game_transitions.append(transition)

                    self.prev_state = curr_state
                    self.prev_score = self.score
                    self.prev_lives = self.lives

                    self.playing()

                elif self.gameState == 'exploding':
                    self.exploding()

                else:
                    self.displayText()
                    self.post_game_reflection()
                    self.initialiseGame()

                pygame.display.flip()

        except (SystemExit, KeyboardInterrupt):
            self.q_agent.save_model()
            print("Model saved on exit")
            pygame.quit()
            sys.exit(0)

    def post_game_reflection(self):
        if not self.game_transitions:
            return

        style_reward = 100 * (self.score // 4000)
        num_transitions = len(self.game_transitions)
        if num_transitions > 0:
            bonus_per_transition = style_reward / num_transitions
            adjusted_transitions = [
                (s, a, r + bonus_per_transition, ns)
                for s, a, r, ns in self.game_transitions
            ]
            self.q_agent.train(adjusted_transitions)
            self.q_agent.check_and_save_model(self.score)  # Added to save high-score model
            self.q_agent.log_game_stats(self.score, self.lives, self.rocks_destroyed, self.shots_fired)
            print(f"Post-game: Score={self.score}, Lives={self.lives}, Rocks={self.rocks_destroyed}, Shots={self.shots_fired}, Style Reward={style_reward}, Shot Accuracy={self.rocks_destroyed / max(self.shots_fired, 1):.3f}")

    def playing(self):
        if self.lives == 0:
            self.gameState = 'attract_mode'
        else:
            self.processKeys()
            self.checkCollisions()
            if len(self.rockList) == 0:
                self.levelUp()

    def doSaucerLogic(self):
        if self.saucer is not None:
            if self.saucer.laps >= 2:
                self.killSaucer()

        if self.secondsCount % 2000 == 0 and self.saucer is None:
            randVal = random.randrange(0, 10)
            if randVal <= 3:
                self.saucer = Saucer(
                    self.stage, Saucer.smallSaucerType, self.ship)
            else:
                self.saucer = Saucer(
                    self.stage, Saucer.largeSaucerType, self.ship)
            self.stage.addSprite(self.saucer)
            # Stage.printSpriteList(self.stage)

    def exploding(self):
        self.explodingCount += 1
        if self.explodingCount > self.explodingTtl:
            self.gameState = 'playing'
            [self.stage.spriteList.remove(debris)
             for debris in self.ship.shipDebrisList]
            self.ship.shipDebrisList = []

            if self.lives == 0:
                self.ship.visible = False
            else:
                self.createNewShip()

    def levelUp(self):
        self.numRocks += 1
        self.createRocks(self.numRocks)

    def displayText(self):
        current_dir = os.path.dirname(__file__)
        res_dir = os.path.join(current_dir, "../res")
        font1 = pygame.font.Font(os.path.join(res_dir, 'Hyperspace.otf'), 50)
        font2 = pygame.font.Font(os.path.join(res_dir, 'Hyperspace.otf'), 20)
        font3 = pygame.font.Font(os.path.join(res_dir, 'Hyperspace.otf'), 30)

        titleText = font1.render('Asteroids', True, (180, 180, 180))
        titleTextRect = titleText.get_rect(centerx=self.stage.width/2)
        titleTextRect.y = self.stage.height/2 - titleTextRect.height*2
        self.stage.screen.blit(titleText, titleTextRect)

        keysText = font2.render(
            '(C) 1979 Atari INC.', True, (255, 255, 255))
        keysTextRect = keysText.get_rect(centerx=self.stage.width/2)
        keysTextRect.y = self.stage.height - keysTextRect.height - 20
        self.stage.screen.blit(keysText, keysTextRect)

        instructionText = font3.render(
            'Press start to Play', True, (200, 200, 200))
        instructionTextRect = instructionText.get_rect(
            centerx=self.stage.width/2)
        instructionTextRect.y = self.stage.height/2 - instructionTextRect.height
        self.stage.screen.blit(instructionText, instructionTextRect)

    def displayScore(self):
        current_dir = os.path.dirname(__file__)
        res_dir = os.path.join(current_dir, "../res")
        font1 = pygame.font.Font(os.path.join(res_dir, 'Hyperspace.otf'), 30)
        scoreStr = str("%02d" % self.score)
        scoreText = font1.render(scoreStr, True, (200, 200, 200))
        scoreTextRect = scoreText.get_rect(centerx=100, centery=45)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def displayPaused(self):
        if self.paused:
            current_dir = os.path.dirname(__file__)
            res_dir = os.path.join(current_dir, "../res")
            font1 = pygame.font.Font(os.path.join(res_dir, 'Hyperspace.otf'), 30)
            pausedText = font1.render("Paused", True, (255, 255, 255))
            textRect = pausedText.get_rect(
                centerx=self.stage.width/2, centery=self.stage.height/2)
            self.stage.screen.blit(pausedText, textRect)
            pygame.display.update()

    def input(self, events):
        self.frameAdvance = False
        for event in events:
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
                if self.gameState == 'playing':
                    if event.key == K_SPACE:
                        self.ship.fireBullet()
                    elif event.key == K_b:
                        self.ship.fireBullet()
                    elif event.key == K_h:
                        self.ship.enterHyperSpace()
                elif self.gameState == 'attract_mode':
                    if event.key == K_RETURN or True:
                        self.initialiseGame()

                if event.key == K_p:
                    self.paused = not self.paused

                if event.key == K_j:
                    self.showingFPS = not self.showingFPS

                if event.key == K_f:
                    pygame.display.toggle_fullscreen()

            elif event.type == KEYUP:
                if event.key == K_o:
                    self.frameAdvance = True

    def processKeys(self):
        key = pygame.key.get_pressed()
        action = {'left': 0, 'right': 0, 'thrust': 0}

        if key[K_LEFT] or key[K_z]:
            self.ship.rotateLeft()
            action['left'] = 1
        elif key[K_RIGHT] or key[K_x]:
            self.ship.rotateRight()
            action['right'] = 1
        if key[K_UP] or key[K_n]:
            self.ship.increaseThrust()
            self.ship.thrustJet.accelerating = True
            action['thrust'] = 1
        else:
            self.ship.thrustJet.accelerating = False

        return action

    def apply_ai_action(self, action_idx):
        self.ship.thrustJet.accelerating = False
        if action_idx == 1:
            self.ship.rotateLeft()
        elif action_idx == 2:
            self.ship.rotateRight()
        elif action_idx == 3:
            self.ship.increaseThrust()
            self.ship.thrustJet.accelerating = True
        elif action_idx == 4:
            self.ship.fireBullet()
            self.shots_fired += 1

    def checkCollisions(self):
        newRocks = []
        shipHit, saucerHit = False, False

        for rock in self.rockList[:]:
            rockHit = False

            if not self.ship.inHyperSpace and rock.collidesWith(self.ship):
                p = rock.checkPolygonCollision(self.ship)
                if p is not None:
                    shipHit = True
                    rockHit = True

            if self.saucer is not None:
                if rock.collidesWith(self.saucer):
                    saucerHit = True
                    rockHit = True

                if self.saucer.bulletCollision(rock):
                    rockHit = True

                if self.ship.bulletCollision(self.saucer):
                    saucerHit = True
                    self.score += self.saucer.scoreValue

            if self.ship.bulletCollision(rock):
                rockHit = True
                self.rocks_destroyed += 1

            if rockHit:
                self.rockList.remove(rock)
                self.stage.removeSprite(rock)

                if rock.rockType == Rock.largeRockType:
                    # playSound("explode1")  # Commented out audio
                    newRockType = Rock.mediumRockType
                    self.score += 50
                elif rock.rockType == Rock.mediumRockType:
                    # playSound("explode2")  # Commented out audio
                    newRockType = Rock.smallRockType
                    self.score += 100
                else:
                    # playSound("explode3")  # Commented out audio
                    self.score += 200

                if rock.rockType != Rock.smallRockType:
                    for _ in range(0, 2):
                        position = Vector2d(rock.position.x, rock.position.y)
                        newRock = Rock(self.stage, position, newRockType)
                        self.stage.addSprite(newRock)
                        self.rockList.append(newRock)

                self.createDebris(rock)

        if self.saucer is not None:
            if not self.ship.inHyperSpace:
                if self.saucer.bulletCollision(self.ship):
                    shipHit = True

                if self.saucer.collidesWith(self.ship):
                    shipHit = True
                    saucerHit = True

            if saucerHit:
                self.createDebris(self.saucer)
                self.killSaucer()

        if shipHit:
            self.killShip()

    def killShip(self):
        # stopSound("thrust")  # Commented out audio
        # playSound("explode2")  # Commented out audio
        self.explodingCount = 0
        self.lives -= 1
        if self.livesList:
            ship = self.livesList.pop()
            self.stage.removeSprite(ship)

        self.stage.removeSprite(self.ship)
        self.stage.removeSprite(self.ship.thrustJet)
        self.gameState = 'exploding'
        self.ship.explode()

    def killSaucer(self):
        # stopSound("lsaucer")  # Commented out audio
        # stopSound("ssaucer")  # Commented out audio
        # playSound("explode2")  # Commented out audio
        self.stage.removeSprite(self.saucer)
        self.saucer = None

    def createDebris(self, sprite):
        for _ in range(0, 25):
            position = Vector2d(sprite.position.x, sprite.position.y)
            debris = Debris(position, self.stage)
            self.stage.addSprite(debris)

    def displayFps(self):
        current_dir = os.path.dirname(__file__)
        res_dir = os.path.join(current_dir, "../res")
        font2 = pygame.font.Font(os.path.join(res_dir, 'Hyperspace.otf'), 15)
        fpsStr = str(self.fps) + ' FPS'
        scoreText = font2.render(fpsStr, True, (255, 255, 255))
        scoreTextRect = scoreText.get_rect(
            centerx=(self.stage.width/2), centery=15)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def displayEpsilon(self):
        font = pygame.font.Font(None, 30)
        eps_text = font.render(f"Epsilon: {self.q_agent.epsilon:.3f}", True, (255, 255, 255))
        self.stage.screen.blit(eps_text, (10, 70))

    def checkScore(self):
        if self.score > 0 and self.score > self.nextLife:
            # playSound("extralife")  # Commented out audio
            self.nextLife += 10000
            self.addLife(self.lives)

if not pygame.font:
    print('Warning, fonts disabled')
# if not pygame.mixer:  # Commented out audio check
#     print('Warning, sound disabled')

game = Asteroids()
game.playGame()