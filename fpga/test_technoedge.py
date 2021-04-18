import re
import os
import copy
import time
import random
import itertools
import warnings
import struct
import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.drivers.amba import AXI4LiteMaster

from bitstring import BitArray

N_DANCERS = 1
DEBUG = True
FIXED_PT_FACTOR = float(1 << 24)

FC1_IN = 84
FC1_OUT = 64
FC2_OUT = 16
FC3_OUT = 3

class TechnoedgeTB(object):

    def __init__(self, dut, debug = False):
        self.dut = dut

        # Get clock up and running
        self.clk = cocotb.fork(Clock(dut.ap_clk, 10, units='ns').start())

        # Initial signal values and setup
        self.dut.ap_rst_n = 1
        
        # AXIL4 Master
        self.axim = AXI4LiteMaster(dut, "s_axi_AXILiteS", dut.ap_clk)
        
        # Map constants
        self.const = self.dut.technoedge_AXILiteS_s_axi_U
        
    async def reset(self, duration = 10):
        # Hold reset for a few cycles
        self.dut.ap_rst_n = 0

        for _ in range(10):
            await RisingEdge(self.dut.ap_clk)

        self.dut.ap_rst_n = 1
        
    async def run(self):
        await self.axim.write(0x00, 1)
        counter = 0
        
        while True:
            for _ in range(1000):
                await RisingEdge(self.dut.ap_clk)
            counter = counter + 1
                
            val = await self.axim.read(0x00)
            if val & 0x2:
                cocotb.log.info("Ran in about %dK clock cycles" % counter)
                return
    
    async def put_input(self, data):
        offset = 0
        for sample in data:
            await self.axim.write(int(self.const.ADDR_SENSOR_DATA_V_BASE) + offset * 4, int(sample))
            offset = offset + 1
      
        
# Pytorch model
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.fc1 = nn.Linear(FC1_IN, FC1_OUT)
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT)
        self.fc3 = nn.Linear(FC2_OUT, FC3_OUT)


def get_model():
    model = DNN()
    model.train()
    model.eval()
    return model

@cocotb.test()
async def test_neural_network(dut):
    tb = TechnoedgeTB(dut)
    await tb.reset()
    
    # Create pytorch model with random weights
    model = get_model()
    
    # Put data
    random_data_raw = torch.randn(FC1_IN)
    random_data = (random_data_raw * FIXED_PT_FACTOR).round()
    await tb.put_input(random_data)
    
    
    # FC1 weight and bias
    weight = model.fc1.weight * FIXED_PT_FACTOR
    for i in range(FC1_OUT):
        for j in range(FC1_IN):
            await tb.axim.write(int(tb.const.ADDR_WTS_FC1_WT_V_BASE) + (i * FC1_IN + j) * 4, int(weight[i][j]))
            
    bias = model.fc1.bias * FIXED_PT_FACTOR
    for i in range(FC1_OUT):
        await tb.axim.write(int(tb.const.ADDR_WTS_FC1_BIAS_V_BASE) + i * 4, int(bias[i])) 
        
        
    # FC2 weight and bias
    weight = model.fc2.weight * FIXED_PT_FACTOR
    for i in range(FC2_OUT):
        for j in range(FC1_OUT):
            await tb.axim.write(int(tb.const.ADDR_WTS_FC2_WT_V_BASE) + (i * FC1_OUT + j) * 4, int(weight[i][j]))
            
    bias = model.fc2.bias * FIXED_PT_FACTOR
    for i in range(FC2_OUT):
        await tb.axim.write(int(tb.const.ADDR_WTS_FC2_BIAS_V_BASE) + i * 4, int(bias[i])) 
        
        
    # FC3 weight and bias
    weight = model.fc3.weight * FIXED_PT_FACTOR
    for i in range(FC3_OUT):
        for j in range(FC2_OUT):
            await tb.axim.write(int(tb.const.ADDR_WTS_FC3_WT_V_BASE) + (i * FC2_OUT + j) * 4, int(weight[i][j]))
            
    bias = model.fc3.bias * FIXED_PT_FACTOR
    for i in range(FC3_OUT):
        await tb.axim.write(int(tb.const.ADDR_WTS_FC3_BIAS_V_BASE) + i * 4, int(bias[i])) 
        
        
    # Run neural network
    await tb.run()

    exp_out = model.fc1(random_data_raw)
    exp_out = model.fc2(exp_out)
    exp_out = model.fc3(exp_out)
    
    print(exp_out)
    
    for i in range(FC3_OUT):
        val = await tb.axim.read(int(tb.const.ADDR_RESULT_RESULT_V_BASE) + i * 4)
        val = ctypes.c_int(val).value
        val = val / FIXED_PT_FACTOR
        cocotb.log.info(val)
        
        assert float(exp_out[i]) - val < 0.0001, "Difference too large between %f and %f." % (float(exp_out[i]), val)

    cocotb.log.info("DONE")
