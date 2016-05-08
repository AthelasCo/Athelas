################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/GraphGen_Cuda.cu \

CU_DEPS += \
./src/GraphGen_Cuda.d \

CPP_SRCS += \
../src/Edge.cpp \
../src/GraphGen_notSorted.cpp \
../src/PaRMAT.cpp \
../src/Square.cpp \
../src/utils.cpp 

OBJS += \
./src/Edge.o \
./src/GraphGen_notSorted.o \
./src/GraphGen_Cuda.o \
./src/GraphGen_sorted.o \
./src/PaRMAT.o \
./src/Square.o \
./src/utils.o 

CPP_DEPS += \
./src/Edge.d \
./src/GraphGen_notSorted.d \
./src/GraphGen_Cuda.d \
./src/GraphGen_sorted.d \
./src/PaRMAT.d \
./src/Square.d \
./src/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(CXX)  $(CXXFLAGS) -c -fmessage-length=0 -pthread -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


src/%.o: ../src/%.cu
	$(NVCC) $< $(NVCCFLAGS) -c -I ../src/cub-1.5.2/ -std=c++11 -o $@