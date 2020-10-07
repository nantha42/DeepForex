import torch
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import numpy as np
import time
# from essential_functions import *

import pygame as py

py.init()

class Simulator:
    def __init__(self,GUI=True):
        if GUI:
            self.win = py.display.set_mode((800,600))
        self.exit = False
        self.data = None
        self.profit = 0.0
        self.window_size = 72
        self.window = None
        self.transaction_cost = 5
        self.balance = torch.tensor(0.0).type(torch.float64)
        self.equity = torch.tensor(0.0).type(torch.float64)
        self.start_at = 40
        self.observed_profit = 0
        self.font = py.font.SysFont("comicsansms", 30)
        self.reading_font = py.font.SysFont("comicsansms", 17)
        self.step_count = 0
        self.data_index = self.window_size + 0 + self.start_at
        self.clock = py.time.Clock()
        self.init_variables()
        self.pause = False
        
        
    def init_variables(self):
        self.t1 = time.time()
        self.order_index = None
        self.state = 'N'
        self.buy_position = None
        self.sell_position = None
        self.history_orders = []
    

    def event_handler(self):
        for event in py.event.get():
            if event.type==py.QUIT:
                self.exit = True

            if event.type == py.KEYDOWN:
                if event.key == py.K_SPACE:
                    self.pause = not self.pause

                if event.key == py.K_b:
                    if self.state == 'N':
                        self.state = 'B'

                if event.key == py.K_s:
                    if self.state == 'N':
                        self.state = 'S'

                if event.key == py.K_c:
                    if self.state == 'H':
                        self.state = 'C'

                if event.key == py.K_LEFT:
                    self.data_index -=1
                
                if event.key == py.K_RIGHT:
                    self.data_index += 1

    def set_data(self,file):
        data = pd.read_csv(file)
        open_price = data["Open"]
        high_price = data["High"]
        low_price = data["Low"]
        close_price = data["Close"]
        self.data = [open_price,high_price,low_price,close_price]
        self.data = torch.tensor(self.data)
        self.window = self.data[:,self.data_index-self.window_size:self.data_index]
        # print(self.data)
    
    def update(self):
        # if time.time()-self.t1 >= .1:
        #     # self.win.fill((0,0,0))
        #     self.t1 = time.time()
        if self.data_index < self.data.size(-1)-1:
            self.data_index += 1
            window = self.data[:, self.data_index- self.window_size:self.data_index]
            self.window = window
            # print(self.state)
            if self.state == 'H':
                if self.buy_position != None:
                    self.profit = (self.window[:,-1][0]-self.buy_position)*1e5*0.1 - self.transaction_cost
                if self.sell_position != None:
                    self.profit = (self.sell_position - self.window[:,-1][0])*1e5*0.1 - self.transaction_cost
            

    def draw2(self):
        self.draw()
        if not  self.pause:
            self.data_index += 1
            pass
    def draw(self):
        if time.time()-self.t1 >= .1:
            self.win.fill((0,0,0))
            self.t1 = time.time()
            if self.data_index < self.data.size(-1)-1:
                if not  self.pause:
                    # self.data_index += 1
                    pass
                    
            # original_window_size = self.window_size 
            # self.window_size = 128
            # self.update()
            window = self.data[:, self.data_index- self.window_size:self.data_index]
            self.window = window
            
            maximum = torch.max(window).item()
            minimum = torch.min(window).item()
            # print(maximum,minimum)
            
            vertical_space_available = 400
            window_scale = vertical_space_available/( (maximum-minimum)*1e5 )
            
            space_available = 700
            each_bar_width = space_available//self.window_size
            bar_width = each_bar_width*0.60
            bar_left_space = each_bar_width*0.20
            graph_x = 0
            graph_y = 40
            line = None

            dp = (maximum-minimum)/10

            for i in range(0,10):
                text = self.reading_font.render(str(round(maximum-i*dp,5)) ,10,(255,255,255))
                text_rect = text.get_rect()
                text_rect.topleft = (graph_x+space_available,graph_y+i*(vertical_space_available/10))
                self.win.blit(text,text_rect)

            for i in range(window.size(-1)):
                bar = window[:,i:i+1]
                o,h,l,c = bar
                if o > c:
                    #bear candle
                    bear_candle = py.Rect(i*each_bar_width + bar_left_space, graph_y + (maximum-o)*1e5*window_scale ,bar_width,(o-c)*1e5*window_scale )
                    py.draw.rect(self.win,(255,0,0),bear_candle)
                if o <= c:
                    #bull candle
                    bull_candle = py.Rect( i*each_bar_width + bar_left_space, graph_y + (maximum-c)*1e5*window_scale ,bar_width,( c-o )*1e5*window_scale )
                    py.draw.rect(self.win,(0,255,0),bull_candle)
            
            for order in self.history_orders:
                if order[0] == 'B':
                    x1 = graph_x + (self.window_size - (self.data_index-order[2]))*each_bar_width
                    y1 = graph_y+(maximum - order[1])*window_scale*1e5
                    x2 = graph_x+(self.window_size - (self.data_index-order[4]))*each_bar_width
                    y2 = graph_y+(maximum - order[3])*window_scale*1e5
                    py.draw.line(self.win,(0,0,255),(x1,y1),(x2,y2),2)
                    py.draw.circle(self.win,(0,0,255),(x1,y1),6,6)
                    py.draw.circle(self.win,(255,0,0),(x2,y2),6,6)

                if order[0] == 'S':
                    x1 = graph_x + (self.window_size - (self.data_index-order[2]))*each_bar_width
                    y1 = graph_y+(maximum - order[1])*window_scale*1e5
                    x2 = graph_x+(self.window_size - (self.data_index-order[4]))*each_bar_width
                    y2 = graph_y+(maximum - order[3])*window_scale*1e5
                    py.draw.line(self.win,(255,0,0),(x1,y1),(x2,y2),2)
                    py.draw.circle(self.win,(255,0,0),(x1,y1),6,6)
                    py.draw.circle(self.win,(0,0,255),(x2,y2),6,6)

            if self.state == 'H':
                # order_posx = (window.size(-1) - self.order_index-1)*each_bar_width
                order_posx = (self.window_size - (self.data_index - self.order_index) )*each_bar_width

                order_size = 6
                if self.buy_position != None:
                    order_posy = (maximum - self.buy_position)*1e5*window_scale
                    py.draw.circle(self.win,(0,0,255),(graph_x + order_posx,graph_y+order_posy),order_size,order_size)
                
                if self.sell_position != None:
                    order_posy = (maximum - self.sell_position)*1e5*window_scale
                    py.draw.circle(self.win,(255,0,0),(graph_x + order_posx,graph_y+order_posy),order_size,order_size)
            self.show_balance()
            # self.window_size = original_window_size
            # window = self.data[:, self.data_index- self.window_size:self.data_index]
            # self.window = window

    def manage(self):
        if self.state=='B':
            self.buy_position = self.window[:,-1][0]
            self.order_index = self.data_index
            self.state = 'H'

        if self.state == 'S':
            self.sell_position = self.window[:,-1][0]
            self.order_index = self.data_index
            self.state = 'H'

        if self.state=='C':
            if self.buy_position != None:
                self.balance +=  (self.window[:,-1][0]-self.buy_position)*1e5*0.1 -self.transaction_cost 
                self.history_orders.append(['B',self.buy_position,self.order_index,self.window[:,-1][0],self.data_index])
                self.buy_position = None
                self.state = 'N'
                self.order_index = None
                # print("Balance: ",self.balance)
                # print(self.profit)
                
            if self.sell_position != None:
                self.balance +=  (self.sell_position - self.window[:,-1][0])*1e5*0.1 -self.transaction_cost
                self.history_orders.append(['S',self.sell_position,self.order_index,self.window[:,-1][0],self.data_index])
                # print(self.history_orders)
                self.sell_position = None
                self.order_index = None
                self.state = 'N'
                # print("Balance: ",self.balance)
                # print(self.profit)

    def getcount(self):
        return self.data.size(-1)-1-self.data_index

    def reset(self):
        self.data_index = self.window_size + 0 + self.start_at
        self.balance = torch.tensor(0.0).type(torch.float64)
        self.profit = 0.0
        self.state = 'N'
        self.step_count = 0
        self.sell_position=None
        self.buy_position=None
        self.window = self.data[:,self.data_index-self.window_size:self.data_index]
        self.history_orders = []

    def step(self,action):
        # print(self.step_count+=1)
        reward = 0
        self.step_count +=1

        if action==1:
            #Buy
            if self.state == 'N':
                self.state = 'B'
                self.profit = 0
                self.reward = 1
            
            
        if action == 2:
            #sell
            if self.state == 'N':
                self.state = 'S'
                self.profit = 0
                self.reward = 1
            
        if action == 3:
            #Close
            if self.state == 'H':
                self.state = 'C'
                if self.buy_position != None:
                    # if self.window[:,-1][0]-self.buy_position >0:
                    reward = ((self.window[:,-1][0]-self.buy_position)*1e5)/10
                    # else:
                    #     reward = ((self.window[:,-1][0]-self.buy_position)*1e5)/10

                elif self.sell_position != None:
                    # if  self.sell_position - self.window[:,-1][0]>0:
                    reward = ((self.sell_position-self.window[:,-1][0])*1e5)/10
                    
                    
        
            
        self.update()
        self.manage()
        if self.order_index != None and self.order_index+30 < self.data_index:
            reward = -3
            
        nextstate = 0

        if self.state=='N':nextstate = 0
        elif self.state=='H':nextstate = 1
        return [nextstate,self.profit,reward]
    
    def draw_matrix(self,mat,x,y,mini,maxi):
        sq =4
        da = (maxi-mini)/255.0
        
        for r in range(len(mat)):
            for c in range(len(mat[r])):
                rect = (x + r*sq,y + c*sq,sq,sq)
                v = mat[r][c]
                ci = int((v-mini)/da)
                py.draw.rect(self.win,(ci,ci,0),rect)

    def plot_weights(self,x):
        i = 0
        gx = 0
        gy = 500
        sq = 4

        for weight in x:
            if len(weight.shape) == 2:
                r,c = weight.shape[0],weight.shape[1]
                
                if r > 1:
                    mini,maxi = weight.min().item(),weight.max().item()
                    weight = weight.tolist()
                    self.draw_matrix(weight,gx,gy,mini,maxi)
                    gx = gx + r*sq + 20
                else:
                    mini,maxi = weight.min().item(),weight.max().item()
                    weight = weight.tolist()
                    self.draw_matrix(weight,gx,gy,mini,maxi)
                    gx = gx + r*sq + 20


    def show_balance(self):
        if self.balance.item() < 0:
            color = (200,0,0)
        else: color = (0,200,0)
        text = self.font.render("Bal: "+str( round(self.balance.item(),2) ) ,True,color)
        text_rect = text.get_rect()
        text_rect.topleft = (10,10)
        self.win.blit(text, text_rect)

    def run(self):
        while not self.exit:
            print(self.balance)
            self.event_handler()
            self.manage()
            self.draw2()
            
            if self.pause:
                self.clock.tick(10)
            else:
                self.clock.tick(30)
            py.display.update()
            

if __name__ == '__main__':
    s = Simulator(True)
    s.set_data("../dataset/EURUSD30min2015-17.csv")
    # print(s.step(1))
    
    # print(s.step)
    s.run()