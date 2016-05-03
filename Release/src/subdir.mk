################################################################################
# Automatically-generated file. Do not edit!
################################################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS := 

ifneq ($(wildcard /usr/local/cuda/.*),)
# Building on Latedays
NVCCFLAGS=-O3 -m64 -arch compute_20
LIBS += GL glut cudart
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
else
# Building on Linux
NVCCFLAGS=-O3 -m64 -arch compute_20
LIBS += GL glut cudart
LDFLAGS=-L/usr/local/depot/cuda-6.5/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

CU_FILES   := ../src/cudaRenderer.cu

CU_DEPS    := ./src/GraphGen_notSorted_Cuda.d 

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Edge.cpp \
../src/GraphGen_notSorted.cpp \
../src/GraphGen_sorted.cpp \
../src/PaRMAT.cpp \
../src/Square.cpp \
../src/utils.cpp 

OBJS += \
./src/Edge.o \
./src/GraphGen_notSorted.o \
./src/GraphGen_notSorted_Cuda.o \
./src/GraphGen_sorted.o \
./src/PaRMAT.o \
./src/Square.o \
./src/utils.o 

CPP_DEPS += \
./src/Edge.d \
./src/GraphGen_notSorted.d \
./src/GraphGen_sorted.d \
./src/PaRMAT.d \
./src/Square.d \
./src/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -pthread -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
        $(NVCC) $< $(NVCCFLAGS) -c -o $@
