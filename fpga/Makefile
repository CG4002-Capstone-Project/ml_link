PWD=$(shell pwd)
export PYTHONPATH := $(PWD)/:$(PYTHONPATH)
VERILOG_SOURCES = $(PWD)/hw_acc/syn/verilog/*.v

TOPLEVEL := technoedge
MODULE   := test_technoedge

include $(shell cocotb-config --makefiles)/Makefile.sim
