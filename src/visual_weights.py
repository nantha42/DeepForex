import torch
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import numpy as np
import time

import pygame as py

py.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Simulator:
    def __init__(self, GUI=True):
        if GUI:
            self.win = py.display.set_mode((800, 600))
        self.font = py.font.SysFont("comicsansms", 30)
        self.reading_font = py.font.SysFont("comicsansms", 17)
        self.clock = py.time.Clock()
        self.size = 1
        self.scroll = 0
        self.scroll_down= False
        self.scroll_up = False

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                self.exit = True

            if event.type == py.KEYDOWN:
                if event.key == py.K_a:
                    if self.size <2:
                        self.size = self.size/1.5
                    else:
                        self.size -= 1
                if event.key == py.K_s:
                    if self.size < 2:
                        self.size = self.size * 1.5
                    else:
                        self.size += 1
                if event.key == py.K_DOWN:
                    self.scroll_down = True
                if event.key  == py.K_UP:
                    self.scroll_up = True
            if event.type == py.KEYUP:
                if event.key == py.K_DOWN:
                    self.scroll_down = False
                if event.key  == py.K_UP:
                    self.scroll_up = False

    def draw2(self):
        self.draw()
        if not self.pause:
            self.data_index += 1
            pass

    def draw(self):
        self.clock.tick(10)
        py.display.update()
        pass

    def getcount(self):
        return self.data.size(-1) - 1 - self.data_index

    def draw_model(self,model):
        self.win.fill((0,0,0))
        all_weights = []
        all_gradients = []
        size = self.size
        for p in model.parameters():
            all_weights.append(p)
            all_gradients.append(p.grad)

        if self.scroll_down:
            self.scroll -= 1*size
        if self.scroll_up:
            self.scroll += 1*size

        x,y = 10,30+self.scroll

        for weight,grad in zip(all_weights,all_gradients):
            if len(weight.shape)>1:
                mini, maxi = weight.min().item(), weight.max().item()

                self.draw_matrix(weight,x,y,mini,maxi,size)
                if grad != None:
                    mini, maxi = grad.min().item(), grad.max().item()
                    self.draw_matrix(grad,x+len(weight)*size+2*size,y,mini,maxi,size)
                y += len(weight[0])*size+2*size

            if len(weight.shape) ==1:
                mini, maxi = weight.min().item(), weight.max().item()
                self.draw_matrix(weight.view(-1,1), x, y, mini, maxi, size)
                if grad != None:
                    mini, maxi = grad.min().item(), grad.max().item()
                    self.draw_matrix(grad.view(-1,1),x+len(weight)*size+2*size,y,mini,maxi,size)
                y += size+2*size 

        # for i in range(3*255):
        #     if i <= 255:
        #         py.draw.line(self.win,(0,i,255),(10,i//2),(100,i//2),1)
        #     elif i <= 255*2:
        #         py.draw.line(self.win, (i-255, 255,255), (10, i//2), (100, i//2), 1)
        #     else:
        #         py.draw.line(self.win, (255, 255, 255), (10, i // 2), (100, i // 2), 1)

    def draw_matrix(self, mat, x, y, mini, maxi,size= 2):
        sq = size
        da = (maxi - mini) / (255.0*3)
        for r in range(len(mat)):
            for c in range(len(mat[r])):
                rect = (x + r * sq, y + c * sq, sq, sq)
                v = mat[r][c]
                ci = 0
                if da != 0:
                    ci = int((v - mini) / da)
                    if ci <= 255:
                        py.draw.rect(self.win, (ci,ci,255), rect)
                    elif ci <= 255*2:
                        py.draw.rect(self.win, (ci-255, 255, 255), rect)
                    else:
                        py.draw.rect(self.win, (255,255,255), rect)


    def plot_weight(self,W,x,y,size=2):
        mini, maxi = W.min().item(), W.max().item()
        self.draw_matrix(W,x,y,mini,maxi,size)


    def plot_weights(self, x):
        i = 0
        gx = 0
        gy = 500
        sq = 0.5

        for weight in x:
            if len(weight.shape) == 2:
                r, c = weight.shape[0], weight.shape[1]

                if r > 1:
                    mini, maxi = weight.min().item(), weight.max().item()
                    weight = weight.tolist()
                    self.draw_matrix(weight, gx, gy, mini, maxi)
                    gx = gx + r * sq + 20
                else:
                    mini, maxi = weight.min().item(), weight.max().item()
                    weight = weight.tolist()
                    self.draw_matrix(weight, gx, gy, mini, maxi)
                    gx = gx + r * sq + 20

    def show_balance(self):
        if self.balance.item() < 0:
            color = (200, 0, 0)
        else:
            color = (0, 200, 0)
        text = self.font.render("Bal: " + str(round(self.balance.item(), 2)), True, color)
        text_rect = text.get_rect()
        text_rect.topleft = (10, 10)
        self.win.blit(text, text_rect)
        if self.equity.item() < 0:
            color = (200, 0, 0)
        else:
            color = (0, 200, 0)
        text = self.font.render("Equ: " + str(round(self.equity.item(), 2)), True, color)
        text_rect = text.get_rect()
        text_rect.topleft = (10, 30)
        self.win.blit(text, text_rect)

    def run(self):
        while not self.exit:
            print(self.balance)
            self.event_handler()
            self.manage()
            if self.pause:
                self.clock.tick(10)
            else:
                self.clock.tick(30)
            py.display.update()


